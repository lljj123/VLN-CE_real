#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist


def main():
    rospy.init_node("cmd_vel_relay")
    publisher = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)
    rospy.Subscriber("/cmd_vel", Twist, publisher.publish, queue_size=10)
    rospy.loginfo("Relaying /cmd_vel to /mobile_base/commands/velocity")
    rospy.spin()


if __name__ == "__main__":
    main()
