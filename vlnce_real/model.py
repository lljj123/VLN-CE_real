"""Standalone PyTorch implementation of the VLN-CE CMA inference network.

Only the single-observation recurrent path used by the real robot is retained.
The module deliberately has no imports from Habitat, Habitat-Baselines, Gym,
or TorchVision.  Attribute names match the original policy so the converted
CMA checkpoint is loaded with ``strict=True``.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size, Tensor

from vlnce_real.resnet import DepthEncoder, RGBEncoder


RGB_SIZE = (224, 224)
DEPTH_SIZE = (256, 256)
NUM_ACTIONS = 4
INSTRUCTION_EMBEDDING_SIZE = 50
INSTRUCTION_HIDDEN_SIZE = 128
STATE_HIDDEN_SIZE = 512
RGB_OUTPUT_SIZE = 256
DEPTH_OUTPUT_SIZE = 128
SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")


def tokenize(
    sentence,
    regex=SENTENCE_SPLIT_REGEX,
    keep=("'s"),
    remove=(",", "?"),
) -> List[str]:
    """Tokenize exactly like the R2R Habitat ``VocabDict`` implementation."""

    sentence = sentence.lower()
    for token in keep:
        sentence = sentence.replace(token, " " + token)
    for token in remove:
        sentence = sentence.replace(token, "")
    return [
        token.strip()
        for token in regex.split(sentence)
        if len(token.strip()) > 0
    ]


class Vocabulary:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"

    def __init__(self, word_list: Iterable[str]) -> None:
        self.word_list = list(word_list)
        if self.UNK_TOKEN not in self.word_list:
            self.word_list = [self.UNK_TOKEN] + self.word_list
        self.word_to_index = {
            word: index for index, word in enumerate(self.word_list)
        }
        self.unknown_index = self.word_to_index[self.UNK_TOKEN]
        self.padding_index = self.word_to_index[self.PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.word_list)

    def tokenize_and_index(self, sentence: str) -> List[int]:
        return [
            self.word_to_index.get(token, self.unknown_index)
            for token in tokenize(sentence)
        ]


def encode_instruction(
    word_list: Iterable[str], instruction: str, max_length: int
) -> Tuple[Tensor, Dict[str, int]]:
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("The navigation instruction must not be empty.")
    if max_length <= 0:
        raise ValueError("Instruction length must be positive.")

    vocabulary = Vocabulary(word_list)
    token_ids = vocabulary.tokenize_and_index(instruction)
    if not token_ids:
        raise ValueError("The instruction tokenizer produced no tokens.")

    token_ids = token_ids[:max_length]
    non_padding_length = len(token_ids)
    unknown_count = sum(
        token == vocabulary.unknown_index for token in token_ids
    )
    token_ids.extend(
        [vocabulary.padding_index] * (max_length - non_padding_length)
    )
    return torch.tensor(token_ids, dtype=torch.long), {
        "length": non_padding_length,
        "unknown_count": int(unknown_count),
        "vocab_size": len(vocabulary),
    }


def batch_observation(
    observations: Dict[str, object], device: torch.device
) -> Dict[str, Tensor]:
    """Add a batch dimension and move one robot observation to ``device``."""

    return {
        name: torch.as_tensor(value).unsqueeze(0).to(device)
        for name, value in observations.items()
    }


class InstructionEncoder(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(
            vocab_size, INSTRUCTION_EMBEDDING_SIZE
        )
        self.embedding_layer.weight.requires_grad_(False)
        self.encoder_rnn = nn.LSTM(
            input_size=INSTRUCTION_EMBEDDING_SIZE,
            hidden_size=INSTRUCTION_HIDDEN_SIZE,
            bidirectional=True,
        )

    @property
    def output_size(self) -> int:
        return INSTRUCTION_HIDDEN_SIZE * 2

    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        instruction = observations["instruction"].long()
        instruction = self.embedding_layer(instruction)
        lengths = (instruction != 0.0).long().sum(dim=2)
        lengths = (lengths != 0.0).long().sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            instruction,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        output, _ = self.encoder_rnn(packed)
        return nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True
        )[0].permute(0, 2, 1)


class GRUStateEncoder(nn.Module):
    """Single-frame recurrent state update used by online robot inference."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.num_recurrent_layers = 1
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
        )

    def forward(
        self, x: Tensor, hidden_states: Tensor, masks: Tensor
    ) -> Tuple[Tensor, Tensor]:
        hidden_states = hidden_states.permute(1, 0, 2)
        if x.size(0) != hidden_states.size(1):
            raise ValueError(
                "The real-robot policy only supports one recurrent state "
                "update per observation batch."
            )
        hidden_states = torch.where(
            masks.view(1, -1, 1),
            hidden_states,
            hidden_states.new_zeros(()),
        )
        x, hidden_states = self.rnn(x.unsqueeze(0), hidden_states)
        return x.squeeze(0), hidden_states.permute(1, 0, 2)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(
        self, sample_shape: Size = torch.Size()
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def mode(self) -> Tensor:
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        return CustomFixedCategorical(logits=self.linear(x))


class CMANet(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.instruction_encoder = InstructionEncoder(vocab_size)
        self.depth_encoder = DepthEncoder()
        self.rgb_encoder = RGBEncoder()
        self.prev_action_embedding = nn.Embedding(NUM_ACTIONS + 1, 32)
        self._hidden_size = STATE_HIDDEN_SIZE

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.rgb_encoder.output_shape[0], RGB_OUTPUT_SIZE),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                DEPTH_OUTPUT_SIZE * 4 * 4 + 64 * 4 * 4,
                DEPTH_OUTPUT_SIZE,
            ),
            nn.ReLU(True),
        )
        self.state_encoder = GRUStateEncoder(
            DEPTH_OUTPUT_SIZE
            + RGB_OUTPUT_SIZE
            + self.prev_action_embedding.embedding_dim,
            STATE_HIDDEN_SIZE,
        )

        first_output_size = (
            STATE_HIDDEN_SIZE
            + RGB_OUTPUT_SIZE
            + DEPTH_OUTPUT_SIZE
            + self.instruction_encoder.output_size
        )
        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            STATE_HIDDEN_SIZE // 2 + RGB_OUTPUT_SIZE,
            1,
        )
        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            STATE_HIDDEN_SIZE // 2 + DEPTH_OUTPUT_SIZE,
            1,
        )
        self.state_q = nn.Linear(STATE_HIDDEN_SIZE, STATE_HIDDEN_SIZE // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size,
            STATE_HIDDEN_SIZE // 2,
            1,
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size,
            STATE_HIDDEN_SIZE // 2,
        )
        self.register_buffer(
            "_scale",
            torch.tensor(1.0 / ((STATE_HIDDEN_SIZE // 2) ** 0.5)),
        )
        self.second_state_compress = nn.Sequential(
            nn.Linear(
                first_output_size
                + self.prev_action_embedding.embedding_dim,
                STATE_HIDDEN_SIZE,
            ),
            nn.ReLU(True),
        )
        self.second_state_encoder = GRUStateEncoder(
            STATE_HIDDEN_SIZE, STATE_HIDDEN_SIZE
        )
        self.progress_monitor = nn.Linear(STATE_HIDDEN_SIZE, 1)

    @property
    def output_size(self) -> int:
        return STATE_HIDDEN_SIZE

    @property
    def num_recurrent_layers(self) -> int:
        return (
            self.state_encoder.num_recurrent_layers
            + self.second_state_encoder.num_recurrent_layers
        )

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", query, key)
        if mask is not None:
            logits = logits - mask.float() * 1e8
        attention = F.softmax(logits * self._scale, dim=1)
        return torch.einsum("ni, nci -> nc", attention, value)

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
        detach_state: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        instruction_embedding = self.instruction_encoder(observations)
        depth_embedding = torch.flatten(
            self.depth_encoder(observations), 2
        )
        rgb_embedding = torch.flatten(self.rgb_encoder(observations), 2)
        previous_action_embedding = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        rgb_input = self.rgb_linear(rgb_embedding)
        depth_input = self.depth_linear(depth_embedding)
        state_input = torch.cat(
            [rgb_input, depth_input, previous_action_embedding], dim=1
        )
        state_source = (
            rnn_states.detach() if detach_state else rnn_states
        )
        output_states = state_source.clone()
        state, output_states[:, 0:1] = self.state_encoder(
            state_input, rnn_states[:, 0:1], masks
        )

        text_state_query = self.state_q(state)
        text_state_key = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_query,
            text_state_key,
            instruction_embedding,
            text_mask,
        )

        rgb_key, rgb_value = torch.split(
            self.rgb_kv(rgb_embedding), STATE_HIDDEN_SIZE // 2, dim=1
        )
        depth_key, depth_value = torch.split(
            self.depth_kv(depth_embedding), STATE_HIDDEN_SIZE // 2, dim=1
        )
        text_query = self.text_q(text_embedding)
        attended_rgb = self._attn(text_query, rgb_key, rgb_value)
        attended_depth = self._attn(
            text_query, depth_key, depth_value
        )

        output = torch.cat(
            [
                state,
                text_embedding,
                attended_rgb,
                attended_depth,
                previous_action_embedding,
            ],
            dim=1,
        )
        output = self.second_state_compress(output)
        output, output_states[:, 1:2] = self.second_state_encoder(
            output, rnn_states[:, 1:2], masks
        )
        return output, output_states


class CMAPolicy(nn.Module):
    def __init__(self, vocab_size: int, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        if num_actions != NUM_ACTIONS:
            raise ValueError(
                "This checkpoint expects exactly {} actions.".format(
                    NUM_ACTIONS
                )
            )
        self.net = CMANet(vocab_size)
        self.action_distribution = CategoricalNet(
            self.net.output_size, num_actions
        )

    def act(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        features, rnn_states = self.net(
            observations, rnn_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        action = (
            distribution.mode()
            if deterministic
            else distribution.sample()
        )
        return action, rnn_states
