# user_command 开机自启动（systemd）

这个目录内已经提供了自启动相关文件。把整个 `user_command` 目录复制到机器人侧后，执行下面步骤。

## 1. 先确认工作空间已构建

```bash
cd /root/udp_ws
colcon build --packages-select user_command
source install/setup.bash
```

## 2. 安装 systemd 服务

```bash
cd /root/udp_ws/src/user_command
sudo bash scripts/install_user_command_service.sh --ws /root/udp_ws --enable-now
```

默认配置：
- service 名称：`user-command.service`
- 网口检查：`eth1` 上存在 `192.168.19.*`
- `ROS_DOMAIN_ID=88`

## 3. 查看状态与日志

```bash
sudo systemctl status user-command.service
sudo journalctl -u user-command.service -f
```

## 4. 常用管理命令

```bash
sudo systemctl restart user-command.service
sudo systemctl stop user-command.service
sudo systemctl disable user-command.service
```

## 5. 可改参数

安装时可以通过参数覆盖默认值，例如：

```bash
sudo bash scripts/install_user_command_service.sh \
  --ws /root/udp_ws \
  --ros-domain-id 88 \
  --iface eth1 \
  --ip-prefix 192.168.19. \
  --enable-now
```

