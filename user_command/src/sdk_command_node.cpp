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

#include "sdk_command/sdk_command_node.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

namespace
{
uint32_t readNetworkU32(const uint8_t * data)
{
  uint32_t value = 0U;
  std::memcpy(&value, data, sizeof(value));
  return ntohl(value);
}

void writeNetworkU32(uint8_t * data, uint32_t value)
{
  const uint32_t network_value = htonl(value);
  std::memcpy(data, &network_value, sizeof(network_value));
}
}  // namespace

SDKCmdNode::SDKCmdNode(const rclcpp::NodeOptions & options)
: Node("user_command_node", options)
{
  this->get_parameter("local_ip", local_ip_);
  this->get_parameter("local_port", local_port_);
  this->get_parameter("sender_ip", sender_ip_);
  this->get_parameter("sender_port", sender_port_);
  this->get_parameter("pub_freq", pub_freq_);
  this->get_parameter("heartbeat_hz", heartbeat_hz_);
  this->get_parameter("cmd_timeout_sec", cmd_timeout_sec_);
  this->get_parameter("body_height", body_height_);
  this->get_parameter("fsm_mode", fsm_mode_);
  this->get_parameter("require_init_packet", require_init_packet_);
  this->get_parameter("enable_cmd_limit", enable_cmd_limit_);
  this->get_parameter("max_linear_speed", max_linear_speed_);
  this->get_parameter("max_angular_speed", max_angular_speed_);

  if (pub_freq_ <= 0.0) {
    pub_freq_ = 160.0;
    RCLCPP_WARN(this->get_logger(), "Invalid pub_freq, fallback to 160.0 Hz.");
  }
  if (heartbeat_hz_ <= 0.0) {
    heartbeat_hz_ = 2.0;
    RCLCPP_WARN(this->get_logger(), "Invalid heartbeat_hz, fallback to 2.0 Hz.");
  }
  if (cmd_timeout_sec_ <= 0.0) {
    cmd_timeout_sec_ = 0.3;
    RCLCPP_WARN(this->get_logger(), "Invalid cmd_timeout_sec, fallback to 0.3 s.");
  }

  user_sdk_publisher_ = this->create_publisher<tita_locomotion_interfaces::msg::LocomotionCmd>(
    tita_topic::user_command, 10);

  if (!initUdpSocket()) {
    throw std::runtime_error("Failed to initialize UDP socket for SDKCmdNode.");
  }

  latest_cmd_time_ = std::chrono::steady_clock::now();
  running_.store(true);
  recv_thread_ = std::thread(&SDKCmdNode::recvLoop, this);

  auto heartbeat_period = std::chrono::duration<double>(1.0 / heartbeat_hz_);
  heartbeat_timer_ = this->create_wall_timer(
    heartbeat_period, std::bind(&SDKCmdNode::heartbeatCallback, this));

  auto period = std::chrono::duration<double>(1.0 / pub_freq_);
  timer_ = this->create_wall_timer(period, std::bind(&SDKCmdNode::timerCallback, this));

  RCLCPP_INFO(
    this->get_logger(),
    "SDK UDP receiver started. listen=%s:%d sender=%s:%d pub=%.1fHz timeout=%.2fs mode=%s",
    local_ip_.c_str(), local_port_,
    sender_ip_.empty() ? "<unset>" : sender_ip_.c_str(),
    sender_port_, pub_freq_, cmd_timeout_sec_, fsm_mode_.c_str());
}

SDKCmdNode::~SDKCmdNode()
{
  running_.store(false);
  if (socket_fd_ >= 0) {
    close(socket_fd_);
    socket_fd_ = -1;
  }

  if (recv_thread_.joinable()) {
    recv_thread_.join();
  }
}

bool SDKCmdNode::initUdpSocket()
{
  socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ < 0) {
    RCLCPP_ERROR(this->get_logger(), "Create UDP socket failed: %s", std::strerror(errno));
    return false;
  }

  int reuse = 1;
  if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    RCLCPP_WARN(this->get_logger(), "setsockopt(SO_REUSEADDR) failed: %s", std::strerror(errno));
  }

  sockaddr_in local_addr {};
  local_addr.sin_family = AF_INET;
  local_addr.sin_port = htons(static_cast<uint16_t>(local_port_));
  if (inet_pton(AF_INET, local_ip_.c_str(), &local_addr.sin_addr) <= 0) {
    RCLCPP_ERROR(this->get_logger(), "Invalid local_ip: %s", local_ip_.c_str());
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  if (bind(socket_fd_, reinterpret_cast<sockaddr *>(&local_addr), sizeof(local_addr)) < 0) {
    RCLCPP_ERROR(this->get_logger(), "Bind UDP socket failed: %s", std::strerror(errno));
    close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  if (!sender_ip_.empty()) {
    sockaddr_in sender_addr {};
    sender_addr.sin_family = AF_INET;
    sender_addr.sin_port = htons(static_cast<uint16_t>(sender_port_));
    if (inet_pton(AF_INET, sender_ip_.c_str(), &sender_addr.sin_addr) <= 0) {
      RCLCPP_ERROR(this->get_logger(), "Invalid sender_ip: %s", sender_ip_.c_str());
      close(socket_fd_);
      socket_fd_ = -1;
      return false;
    }
    sender_addr_valid_ = true;
  } else {
    RCLCPP_WARN(
      this->get_logger(),
      "sender_ip is empty, heartbeat will be disabled. Set sender_ip to receive server-side CMD_VEL.");
  }

  return true;
}

void SDKCmdNode::heartbeatCallback()
{
  if (socket_fd_ < 0 || !sender_addr_valid_) {
    return;
  }

  constexpr size_t data_size = 1U;
  constexpr size_t packet_size = 4U + 4U + data_size + 4U + 4U;
  std::array<uint8_t, packet_size> buffer {};

  writeNetworkU32(buffer.data(), kFrameHeader);
  writeNetworkU32(buffer.data() + 4, data_size);
  buffer[8] = kMsgTypeHeartbeat;
  const uint32_t crc = calculateCrc32(buffer.data() + 8, data_size);
  writeNetworkU32(buffer.data() + 8 + data_size, crc);
  writeNetworkU32(buffer.data() + 8 + data_size + 4, kFrameTail);

  sockaddr_in sender_addr {};
  sender_addr.sin_family = AF_INET;
  sender_addr.sin_port = htons(static_cast<uint16_t>(sender_port_));
  inet_pton(AF_INET, sender_ip_.c_str(), &sender_addr.sin_addr);

  const ssize_t send_len = sendto(
    socket_fd_, buffer.data(), buffer.size(), 0,
    reinterpret_cast<const sockaddr *>(&sender_addr), sizeof(sender_addr));

  if (send_len < 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Send heartbeat failed: %s", std::strerror(errno));
  }
}

void SDKCmdNode::recvLoop()
{
  std::array<uint8_t, 1024> recv_buffer {};

  while (running_.load()) {
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(socket_fd_, &readfds);

    timeval timeout {};
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;

    const int activity = select(socket_fd_ + 1, &readfds, nullptr, nullptr, &timeout);
    if (activity <= 0) {
      continue;
    }

    sockaddr_in remote_addr {};
    socklen_t remote_len = sizeof(remote_addr);
    const ssize_t recv_len = recvfrom(
      socket_fd_, recv_buffer.data(), recv_buffer.size(), 0,
      reinterpret_cast<sockaddr *>(&remote_addr), &remote_len);

    if (recv_len <= 0) {
      continue;
    }

    uint8_t msg_type = 0U;
    std::array<float, 6> vel {};
    if (!parsePacket(recv_buffer.data(), static_cast<size_t>(recv_len), msg_type, vel)) {
      continue;
    }

    if (msg_type == kMsgTypeInit) {
      received_init_ = true;
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Received INIT packet from %s:%u", inet_ntoa(remote_addr.sin_addr), ntohs(remote_addr.sin_port));
      continue;
    }

    if (msg_type != kMsgTypeCmdVel) {
      continue;
    }

    if (enable_cmd_limit_) {
      vel[0] = std::clamp(vel[0], -static_cast<float>(max_linear_speed_), static_cast<float>(max_linear_speed_));
      vel[5] =
        std::clamp(vel[5], -static_cast<float>(max_angular_speed_), static_cast<float>(max_angular_speed_));
    }

    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      latest_vel_ = vel;
      latest_cmd_time_ = std::chrono::steady_clock::now();
      has_valid_cmd_ = true;
    }
  }
}

void SDKCmdNode::timerCallback()
{
  std::array<float, 6> vel {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  bool has_valid_cmd = false;
  std::chrono::steady_clock::time_point cmd_time;

  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    has_valid_cmd = has_valid_cmd_;
    cmd_time = latest_cmd_time_;
    if (has_valid_cmd) {
      vel = latest_vel_;
    }
  }

  if (has_valid_cmd) {
    const auto now = std::chrono::steady_clock::now();
    const auto cmd_age = std::chrono::duration<double>(now - cmd_time).count();
    if (cmd_age > cmd_timeout_sec_) {
      vel = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  if (require_init_packet_ && !received_init_) {
    vel = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }

  publishVelCommand(vel);
}

void SDKCmdNode::publishVelCommand(const std::array<float, 6> & vel)
{
  auto msg = tita_locomotion_interfaces::msg::LocomotionCmd();
  msg.header.stamp = this->now();
  msg.header.frame_id = "cmd";
  msg.fsm_mode = fsm_mode_;

  msg.pose.position.z = body_height_;
  msg.pose.orientation.w = 1.0;

  msg.twist.linear.x = vel[0];
  msg.twist.linear.y = vel[1];
  msg.twist.linear.z = vel[2];
  msg.twist.angular.x = vel[3];
  msg.twist.angular.y = vel[4];
  msg.twist.angular.z = vel[5];

  user_sdk_publisher_->publish(msg);
}

uint32_t SDKCmdNode::calculateCrc32(const uint8_t * data, size_t length)
{
  static const uint32_t crc_table[256] = {
    0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535,
    0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD,
    0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D,
    0x6DDDE4EB, 0xF4D4B551, 0x83D385C7, 0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC,
    0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4,
    0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
    0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59, 0x26D930AC,
    0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
    0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB,
    0xB6662D3D, 0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F,
    0x9FBFE4A5, 0xE8B8D433, 0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB,
    0x086D3D2D, 0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
    0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA,
    0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65, 0x4DB26158, 0x3AB551CE,
    0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A,
    0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
    0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409,
    0xCE61E49F, 0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
    0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739,
    0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8,
    0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1, 0xF00F9344, 0x8708A3D2, 0x1E01F268,
    0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0,
    0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8,
    0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
    0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF,
    0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703,
    0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7,
    0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D, 0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A,
    0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE,
    0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
    0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777, 0x88085AE6,
    0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
    0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D,
    0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5,
    0x47B2CF7F, 0x30B5FFE9, 0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605,
    0xCDD70693, 0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
    0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D};
  uint32_t crc = 0xFFFFFFFFU;

  for (size_t i = 0; i < length; ++i) {
    crc = (crc >> 8U) ^ crc_table[(crc ^ data[i]) & 0xFFU];
  }

  return crc ^ 0xFFFFFFFFU;
}

bool SDKCmdNode::parsePacket(
  const uint8_t * buffer, size_t recv_len, uint8_t & msg_type, std::array<float, 6> & vel) const
{
  constexpr size_t kMinPacketLen = 4U + 4U + 1U + 4U + 4U;
  if (recv_len < kMinPacketLen) {
    return false;
  }

  const uint32_t header = readNetworkU32(buffer);
  const uint32_t data_len = readNetworkU32(buffer + 4);
  const uint32_t tail = readNetworkU32(buffer + recv_len - 4);
  if (header != kFrameHeader || tail != kFrameTail) {
    return false;
  }

  const size_t expected_len = 4U + 4U + static_cast<size_t>(data_len) + 4U + 4U;
  if (expected_len != recv_len || data_len < 1U) {
    return false;
  }

  msg_type = buffer[8];
  const uint32_t recv_crc = readNetworkU32(buffer + 8 + data_len);
  const uint32_t calc_crc = calculateCrc32(buffer + 8, data_len);
  if (recv_crc != calc_crc) {
    return false;
  }

  if (msg_type == kMsgTypeCmdVel || msg_type == kMsgTypeInit) {
    constexpr uint32_t cmd_data_len = 1U + sizeof(float) * 6U;
    if (data_len != cmd_data_len) {
      return false;
    }
    std::memcpy(vel.data(), buffer + 9, sizeof(float) * 6U);
  } else if (msg_type == kMsgTypeHeartbeat) {
    if (data_len != 1U) {
      return false;
    }
  }

  return true;
}
