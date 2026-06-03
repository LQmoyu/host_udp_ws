#!/bin/bash

cd /home/admin/workspace/host_udp_ws || exit 1
source install/setup.bash

ros2 launch dog_udp_comm person_follow_udp_launch.py
