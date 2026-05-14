// Copyright (c) 2023 Direct Drive Technology Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SDK_COMMAND__SDK_COMMAND_NODE_HPP_
#define SDK_COMMAND__SDK_COMMAND_NODE_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "tita_locomotion_interfaces/msg/locomotion_cmd.hpp"
#include "tita_utils/topic_names.hpp"

class SDKCmdNode : public rclcpp::Node
{
public:
  explicit SDKCmdNode(const rclcpp::NodeOptions & options);
  ~SDKCmdNode() override;

private:
  static constexpr uint32_t kFrameHeader = 0xA5A5A5A5;
  static constexpr uint32_t kFrameTail = 0x5A5A5A5A;
  static constexpr uint8_t kMsgTypeCmdVel = 0x01;
  static constexpr uint8_t kMsgTypeHeartbeat = 0x02;
  static constexpr uint8_t kMsgTypeInit = 0x03;

  bool initUdpSocket();
  void recvLoop();
  void heartbeatCallback();
  void timerCallback();
  static uint32_t calculateCrc32(const uint8_t * data, size_t length);
  bool parsePacket(
    const uint8_t * buffer, size_t recv_len, uint8_t & msg_type, std::array<float, 6> & vel) const;
  void publishVelCommand(const std::array<float, 6> & vel);

  rclcpp::Publisher<tita_locomotion_interfaces::msg::LocomotionCmd>::SharedPtr user_sdk_publisher_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::thread recv_thread_;
  std::atomic<bool> running_{false};

  int socket_fd_ = -1;
  bool sender_addr_valid_ = false;

  std::mutex cmd_mutex_;
  std::array<float, 6> latest_vel_{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  std::chrono::steady_clock::time_point latest_cmd_time_;
  bool has_valid_cmd_ = false;
  bool received_init_ = false;

  double pub_freq_ = 160.0;
  std::string local_ip_ = "0.0.0.0";
  int local_port_ = 8890;
  std::string sender_ip_ = "";
  int sender_port_ = 8888;
  double heartbeat_hz_ = 2.0;
  double cmd_timeout_sec_ = 0.3;
  double body_height_ = 0.2;
  std::string fsm_mode_ = "transform_up";
  bool require_init_packet_ = false;
  bool enable_cmd_limit_ = true;
  double max_linear_speed_ = 3.0;
  double max_angular_speed_ = 6.0;
};

#endif  // SDK_COMMAND__SDK_COMMAND_NODE_HPP_
