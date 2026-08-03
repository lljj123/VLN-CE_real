cmake_minimum_required(VERSION 3.0.2)
project(robot_sim_bundle)
find_package(catkin REQUIRED)
catkin_package()

install(DIRECTORY launch lib meshes urdf worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS)
install(PROGRAMS scripts/cmd_vel_relay.py check.sh setup_env.sh start_spawn.sh start_world.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
