cmake_minimum_required(VERSION 3.8)
project(dog_udp_comm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# 编译 C++ 节点
add_executable(sender_node src/sender_node.cpp)
ament_target_dependencies(sender_node rclcpp geometry_msgs std_srvs)

# 安装 C++ 节点
install(TARGETS
  sender_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装 Python 脚本 (赋予可执行权限)
install(PROGRAMS
  scripts/keyboard_teleop.py
  scripts/host_mpc_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 文件
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
