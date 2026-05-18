#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <ctime>
#include <algorithm>

// 协议宏定义
constexpr uint32_t FRAME_HEADER = 0xA5A5A5A5;
constexpr uint32_t FRAME_TAIL = 0x5A5A5A5A;
enum MsgType : uint8_t {
    MSGTYPE_CMD_VEL = 0x01,
    MSGTYPE_HEARTBEAT = 0x02,
    MSGTYPE_INIT = 0x03,
};

struct ClientInfo {
    sockaddr_in addr;
    time_t last_heartbeat;
};

namespace {
uint32_t read_network_u32(const uint8_t *data) {
    uint32_t value = 0U;
    std::memcpy(&value, data, sizeof(value));
    return ntohl(value);
}
}  // namespace

class UdpSenderNode : public rclcpp::Node {
public:
    UdpSenderNode() : Node("udp_cmd_vel_server") {
        // 1. 声明并获取参数
        this->declare_parameter<std::string>("local_ip", "0.0.0.0");
        this->declare_parameter<int>("local_port", 8888);
        this->declare_parameter<bool>("use_fixed_receiver", false);
        this->declare_parameter<std::string>("remote_ip", "");
        this->declare_parameter<int>("remote_port", 0);
        
        std::string local_ip;
        int local_port;
        bool use_fixed_receiver;
        std::string remote_ip;
        int remote_port;
        this->get_parameter("local_ip", local_ip);
        this->get_parameter("local_port", local_port);
        this->get_parameter("use_fixed_receiver", use_fixed_receiver);
        this->get_parameter("remote_ip", remote_ip);
        this->get_parameter("remote_port", remote_port);

        // 2. 初始化 UDP Socket
        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            RCLCPP_FATAL(this->get_logger(), "Socket Create Failed");
            rclcpp::shutdown();
            return;
        }

        sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = inet_addr(local_ip.c_str());
        server_addr.sin_port = htons(static_cast<uint16_t>(local_port));

        if (bind(sockfd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            RCLCPP_FATAL(this->get_logger(), "Bind Failed");
            close(sockfd_);
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(this->get_logger(), "UDP SERVER Started, listening: %s:%d", local_ip.c_str(), local_port);

        use_fixed_receiver_ = use_fixed_receiver;
        if (use_fixed_receiver_) {
            if (remote_ip.empty() || remote_port <= 0 || remote_port > 65535) {
                RCLCPP_FATAL(
                    this->get_logger(),
                    "use_fixed_receiver=true but remote_ip/remote_port invalid (remote_ip='%s', remote_port=%d)",
                    remote_ip.c_str(), remote_port);
                close(sockfd_);
                rclcpp::shutdown();
                return;
            }
            memset(&fixed_client_addr_, 0, sizeof(fixed_client_addr_));
            fixed_client_addr_.sin_family = AF_INET;
            fixed_client_addr_.sin_addr.s_addr = inet_addr(remote_ip.c_str());
            fixed_client_addr_.sin_port = htons(static_cast<uint16_t>(remote_port));

            if (fixed_client_addr_.sin_addr.s_addr == INADDR_NONE) {
                RCLCPP_FATAL(this->get_logger(), "Invalid remote_ip: %s", remote_ip.c_str());
                close(sockfd_);
                rclcpp::shutdown();
                return;
            }
            RCLCPP_INFO(
                this->get_logger(),
                "Fixed receiver enabled: %s:%d (heartbeat registration not required)",
                remote_ip.c_str(), remote_port);
        } else {
            RCLCPP_INFO(this->get_logger(), "Dynamic receiver mode: waiting for heartbeat registration");
        }

        // 3. 创建订阅者和服务端
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/track_cmd_vel", 1, std::bind(&UdpSenderNode::send_cmd_vel, this, std::placeholders::_1));
            
        init_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/init_udp_receiver", std::bind(&UdpSenderNode::init_udp_receiver_handler, this, std::placeholders::_1, std::placeholders::_2));

        // 4. 开启心跳监听线程
        heartbeat_thread_ = std::thread(&UdpSenderNode::handle_heartbeat, this);
    }

    ~UdpSenderNode() {
        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.detach(); // 
        }
        close(sockfd_);
    }

private:
    int sockfd_;
    std::vector<ClientInfo> clients_;
    std::mutex clients_mutex_;
    std::thread heartbeat_thread_;
    bool use_fixed_receiver_{false};
    sockaddr_in fixed_client_addr_{};
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr init_srv_;

    // CRC32 校验 
    uint32_t calculate_crc32(const uint8_t *data, size_t length) {
        uint32_t crc = 0xFFFFFFFFU;
        for (size_t i = 0; i < length; ++i) {
            crc ^= static_cast<uint32_t>(data[i]);
            for (int bit = 0; bit < 8; ++bit) {
                if (crc & 1U) {
                    crc = (crc >> 1U) ^ 0xEDB88320U;
                } else {
                    crc >>= 1U;
                }
            }
        }
        return crc ^ 0xFFFFFFFFU;
    }

    void handle_heartbeat() {
        uint8_t buffer[1024];
        sockaddr_in client_addr;

        while (rclcpp::ok()) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(sockfd_, &readfds);

            timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000;

            int activity = select(sockfd_ + 1, &readfds, nullptr, nullptr, &timeout);
            if (activity <= 0) continue;

            memset(buffer, 0, sizeof(buffer));
            socklen_t client_addr_len = sizeof(client_addr);
            int recv_len = recvfrom(sockfd_, buffer, sizeof(buffer), 0,
                                    (struct sockaddr *)&client_addr, &client_addr_len);
            if (recv_len < 17) {
                continue;
            }

            const uint32_t header = read_network_u32(buffer);
            const uint32_t data_length = read_network_u32(buffer + 4);
            const uint8_t msg_type = buffer[8];
            const uint32_t tail = read_network_u32(buffer + recv_len - 4);

            if (header != FRAME_HEADER || tail != FRAME_TAIL) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "Drop UDP packet from %s:%u (header/tail mismatch, recv_len=%d)",
                    inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port), recv_len);
                continue;
            }

            if (msg_type != MSGTYPE_HEARTBEAT) {
                continue;
            }

            constexpr int kHeartbeatPacketLen = 4 + 4 + 1 + 4 + 4;
            if (data_length != 1 || recv_len != kHeartbeatPacketLen) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "Drop heartbeat from %s:%u (data_length=%u, recv_len=%d)",
                    inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port), data_length, recv_len);
                continue;
            }

            const uint32_t received_crc = read_network_u32(buffer + 9);
            const uint32_t calculated_crc = calculate_crc32(buffer + 8, 1);
            if (received_crc != calculated_crc) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "Drop heartbeat from %s:%u (crc mismatch recv=0x%08x calc=0x%08x msg_type=0x%02x raw=%02x %02x %02x %02x %02x %02x %02x %02x %02x)",
                    inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port),
                    received_crc, calculated_crc, buffer[8],
                    buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7], buffer[8]);
                continue;
            }

            std::lock_guard<std::mutex> lock(clients_mutex_);
            bool found = false;
            for (auto &client : clients_) {
                if (client.addr.sin_addr.s_addr == client_addr.sin_addr.s_addr &&
                    client.addr.sin_port == client_addr.sin_port) {
                    client.last_heartbeat = time(nullptr);
                    found = true;
                    break;
                }
            }
            if (!found) {
                ClientInfo new_client;
                new_client.addr = client_addr;
                new_client.last_heartbeat = time(nullptr);
                clients_.push_back(new_client);
                RCLCPP_INFO(this->get_logger(), "New Client Connected: %s:%d, Current Client Num: %zu",
                            inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port), clients_.size());
            }
        }
    }

    void send_cmd_vel(const geometry_msgs::msg::Twist::SharedPtr msg) {
        const size_t data_size = 1 + sizeof(float) * 6;
        const size_t packet_size = 4 + 4 + data_size + 4 + 4;
        uint8_t buffer[packet_size];

        *((uint32_t *)buffer) = htonl(FRAME_HEADER);
        *((uint32_t *)(buffer + 4)) = htonl(data_size);
        buffer[8] = MSGTYPE_CMD_VEL;
        
        float vel_cmd[6] = {
            static_cast<float>(msg->linear.x), static_cast<float>(msg->linear.y), static_cast<float>(msg->linear.z),
            static_cast<float>(msg->angular.x), static_cast<float>(msg->angular.y), static_cast<float>(msg->angular.z)
        };
        memcpy(buffer + 9, vel_cmd, sizeof(vel_cmd));

        uint32_t crc = calculate_crc32(buffer + 8, data_size);
        *((uint32_t *)(buffer + 8 + data_size)) = htonl(crc);
        *((uint32_t *)(buffer + 8 + data_size + 4)) = htonl(FRAME_TAIL);

        int sent_count = 0;
        if (use_fixed_receiver_) {
            const auto ret = sendto(sockfd_, buffer, packet_size, 0,
                                    (struct sockaddr *)&fixed_client_addr_, sizeof(fixed_client_addr_));
            if (ret >= 0) {
                sent_count = 1;
            }
        } else {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            time_t current_time = time(nullptr);
            clients_.erase(
                std::remove_if(clients_.begin(), clients_.end(),
                               [current_time](const ClientInfo &client) { return (current_time - client.last_heartbeat) > 3; }),
                clients_.end());

            for (const auto &client : clients_) {
                const auto ret = sendto(sockfd_, buffer, packet_size, 0, (struct sockaddr *)&client.addr, sizeof(client.addr));
                if (ret >= 0) {
                    ++sent_count;
                }
            }
        }

        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "cmd_vel sent: vx=%.3f wz=%.3f, receivers=%d",
            msg->linear.x, msg->angular.z, sent_count);
    }

    void init_udp_receiver_handler(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                   std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        (void)req; // 消除未使用警告
        const size_t data_size = 1 + sizeof(float) * 6;
        const size_t packet_size = 4 + 4 + data_size + 4 + 4;
        uint8_t buffer[packet_size];

        *((uint32_t *)buffer) = htonl(FRAME_HEADER);
        *((uint32_t *)(buffer + 4)) = htonl(data_size);
        buffer[8] = MSGTYPE_INIT;
        float vel_cmd[6] = {0, 0, 0, 0, 0, 0};
        memcpy(buffer + 9, vel_cmd, sizeof(vel_cmd));

        uint32_t crc = calculate_crc32(buffer + 8, data_size);
        *((uint32_t *)(buffer + 8 + data_size)) = htonl(crc);
        *((uint32_t *)(buffer + 8 + data_size + 4)) = htonl(FRAME_TAIL);

        int sent_count = 0;
        if (use_fixed_receiver_) {
            const auto ret = sendto(sockfd_, buffer, packet_size, 0,
                                    (struct sockaddr *)&fixed_client_addr_, sizeof(fixed_client_addr_));
            if (ret >= 0) {
                sent_count = 1;
            }
        } else {
            std::lock_guard<std::mutex> glock(clients_mutex_);
            time_t current_time = time(nullptr);
            clients_.erase(
                std::remove_if(clients_.begin(), clients_.end(),
                               [current_time](const ClientInfo &client) { return (current_time - client.last_heartbeat) > 3; }),
                clients_.end());

            for (const auto &client : clients_) {
                const auto ret = sendto(sockfd_, buffer, packet_size, 0, (struct sockaddr *)&client.addr, sizeof(client.addr));
                if (ret >= 0) {
                    ++sent_count;
                }
            }
        }

        res->success = true;
        res->message = "success, receivers=" + std::to_string(sent_count);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UdpSenderNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
