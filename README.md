# face_recognition_ros 人脸识别系统

本项目基于 ROS Noetic、Intel RealSense D435i、InsightFace 和 Streamlit，
既可以在 ROS 中直接运行人脸识别节点，也可以通过浏览器完成人脸录入、
人脸库管理和实时识别。

本文覆盖以下常见用法：

- 在本机启动相机并进行识别
- 启动 Streamlit 网页
- 从同一局域网内的另一台电脑打开网页
- 通过 SSH 端口转发远程打开网页
- 排查相机、视频流和端口问题

## 项目位置

后续命令默认使用以下目录。如果你的工作空间不在这个位置，请替换为实际路径。

```bash
export FACE_ROS_WS="$HOME/catkin_ws"
export FACE_ROS_PKG="$FACE_ROS_WS/src/face_recognition_ros"
export FACE_WEBAPP="$FACE_ROS_PKG/webapp"
```

主要目录：

```text
catkin_ws/src/face_recognition_ros/
├── config/params.yaml             # ROS 识别参数
├── face_data/<姓名>/*.jpg         # 人脸照片
├── launch/face_recognition.launch # D435i + ROS 识别节点
├── model/*.pkl                    # 人脸特征数据库
├── scripts/face_recognition_node.py
└── webapp/
    ├── app.py                     # Streamlit 入口
    ├── start_webapp.sh            # 一键启动脚本
    └── requirements.txt
```

更详细的网页功能说明见
[`webapp/README.md`](webapp/README.md)。

## 一、最快启动方式

打开一个终端，执行：

```bash
cd "$HOME/catkin_ws/src/face_recognition_ros/webapp"
./start_webapp.sh
```

脚本会完成以下操作：

1. 加载 `/opt/ros/noetic/setup.bash`。
2. 加载 `~/catkin_ws/devel/setup.bash`。
3. 在 `0.0.0.0:8501` 启动 Streamlit，其他电脑也能访问。
4. 首次打开网页时自动启动 RealSense 相机驱动。

终端通常会显示类似下面的地址：

```text
本机访问:   http://localhost:8501
其他电脑:   http://192.168.1.23:8501
监听地址:   0.0.0.0:8501   自动启动 roslaunch: 1
```

在运行程序的电脑上打开 `http://localhost:8501`，然后：

1. 在左侧选择 `📹 实时识别`。
2. 取流方式选择 `ROS 话题（配合 realsense2_camera，推荐）`。
3. 确认彩色话题为 `/camera/color/image_raw`。
4. 点击 `▶️ 开始实时视频`。

> 第一次打开页面时，相机驱动可能需要十几秒才能就绪。

### 启动参数

`start_webapp.sh` 支持两个参数：

```bash
./start_webapp.sh [网页端口] [是否自动启动相机]
```

- 第一个参数是 Streamlit 网页端口，默认 `8501`。
- 第二个参数为 `1` 时，打开网页自动启动相机；为 `0` 时关闭自动启动。

常用命令：

```bash
# 默认：8501 端口，打开网页自动启动 D435i
./start_webapp.sh

# 使用 8601 网页端口，仍然自动启动相机
./start_webapp.sh 8601 1

# 8501 端口，完全手动启动相机
./start_webapp.sh 8501 0
```

这里的网页端口只影响网页。实时视频流默认仍然使用 `8502`，可以在
`📹 实时识别 -> 视频流参数 -> 视频流端口` 中修改。

## 二、分步启动 Streamlit 和相机

如果需要分别查看相机和网页日志，可以使用两个终端。

### 终端 1：启动 D435i 相机

```bash
source /opt/ros/noetic/setup.bash
source "$HOME/catkin_ws/devel/setup.bash"

roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true \
  color_width:=640 \
  color_height:=480 \
  color_fps:=30
```

这个命令只启动相机驱动，识别由 Streamlit 网页完成，内存占用较低，
是网页实时识别的推荐方式。

### 终端 2：启动 Streamlit

```bash
source /opt/ros/noetic/setup.bash
source "$HOME/catkin_ws/devel/setup.bash"

cd "$HOME/catkin_ws/src/face_recognition_ros/webapp"
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

浏览器打开 `http://localhost:8501`，进入 `📹 实时识别`，选择
`ROS 话题`，然后点击 `▶️ 开始实时视频`。

如果 Streamlit 运行时发现相机已经由终端 1 启动，会自动复用现有 ROS
话题，不会重复打开相机。

### 第一次运行或代码更新后

如果工作空间尚未编译，或者修改了 ROS 消息、节点源码，先执行：

```bash
cd "$HOME/catkin_ws"
catkin_make
source devel/setup.bash
```

安装网页依赖：

```bash
cd "$HOME/catkin_ws/src/face_recognition_ros/webapp"
pip3 install -r requirements.txt
```

`requirements.txt` 已包含 Streamlit、InsightFace、ONNX Runtime、
OpenCV、NumPy 和 Pillow。

## 三、不打开网页，直接用 ROS 检测

### RealSense D435i

```bash
source /opt/ros/noetic/setup.bash
source "$HOME/catkin_ws/devel/setup.bash"

roslaunch face_recognition_ros face_recognition.launch vis:=rqt
```

该 launch 文件会同时启动：

- `realsense2_camera`
- `face_recognition_node.py`
- `rqt_image_view`，显示 `/face_recognition/visualization`

没有图形桌面或只关心识别结果时，可以关闭可视化窗口：

```bash
roslaunch face_recognition_ros face_recognition.launch vis:=none
```

查看识别结果和可视化话题：

```bash
rostopic hz /face_recognition/results
rostopic echo /face_recognition/results
rostopic hz /face_recognition/visualization
```

识别阈值、YOLO、质量过滤和投票参数位于
[`config/params.yaml`](config/params.yaml)。

### 仅启动相机，不运行 ROS 识别节点

```bash
source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

常用话题：

| 用途 | 话题 |
| --- | --- |
| 彩色图像 | `/camera/color/image_raw` |
| 对齐后的深度图 | `/camera/aligned_depth_to_color/image_raw` |

检查相机是否持续发布图像：

```bash
rostopic hz /camera/color/image_raw
```

### RealSense 直连，不使用 ROS

网页也支持通过 `pyrealsense2` 直接访问 D435i：

```bash
pip3 install pyrealsense2
```

启动网页时关闭 ROS 相机自动启动：

```bash
cd "$HOME/catkin_ws/src/face_recognition_ros/webapp"
./start_webapp.sh 8501 0
```

在 `📹 实时识别` 页面选择 `RealSense 直连（pyrealsense2，无需 ROS）`。

> 同一个 D435i 不能被两个进程同时打开。使用直连方式前，请先停止
> `realsense2_camera` 或 ROS 识别节点；反之亦然。

## 四、从另一台电脑打开网页

这里假设运行 Streamlit 和连接 D435i 的电脑是“服务器”，另一台电脑是
“客户端”。两台电脑必须位于同一局域网，例如连接同一个路由器或 Wi-Fi。

### 1. 在服务器上启动网页

```bash
cd "$HOME/catkin_ws/src/face_recognition_ros/webapp"
./start_webapp.sh
```

不要关闭这个终端。Streamlit 进程结束后，网页也无法继续访问。

### 2. 查询服务器的局域网 IP

在服务器上执行：

```bash
hostname -I
```

找到形如 `192.168.x.x`、`10.x.x.x` 或 `172.16.x.x` 的地址。例如：

```text
192.168.1.23
```

也可以使用：

```bash
ip -4 addr
ip route
```

### 3. 在客户端浏览器中访问

在另一台电脑的浏览器地址栏输入：

```text
http://192.168.1.23:8501
```

请把示例 IP 替换为服务器实际显示的 IP。不要把 `127.0.0.1`、
`localhost` 或 `0.0.0.0` 当作其他电脑的访问地址：

- `127.0.0.1` / `localhost` 永远表示“当前这台电脑”。
- `0.0.0.0` 是监听地址，不是浏览器应该访问的地址。
- 客户端应使用服务器在局域网中的实际 IP。

进入网页后，`📹 实时识别` 使用的是服务器端 D435i，而不是客户端电脑的
摄像头。客户端只负责显示网页和控制识别。

### 4. 开放必要端口

| 端口 | 用途 |
| --- | --- |
| `8501` | Streamlit 网页 |
| `8502` | 实时识别页面的 MJPEG 视频流 |

如果只打开网页、不使用实时视频，可以只放行 `8501`。要使用实时视频，
还需要放行 `8502`。

Ubuntu 使用 UFW 时：

```bash
sudo ufw status
sudo ufw allow 8501/tcp
sudo ufw allow 8502/tcp
```

如果修改了网页端口或视频流端口，需要放行对应端口。

在 Windows 客户端上测试端口是否可达：

```powershell
Test-NetConnection 192.168.1.23 -Port 8501
Test-NetConnection 192.168.1.23 -Port 8502
```

`TcpTestSucceeded` 应为 `True`。如果为 `False`，通常是下列原因之一：

- 两台电脑不在同一局域网，或连接了启用客户端隔离的访客 Wi-Fi。
- Streamlit 没有运行，或者监听的地址不是 `0.0.0.0`。
- 服务器防火墙阻止了端口。
- 服务器 IP 查询错误，选到了虚拟机、Docker 或 VPN 网卡的地址。
- 中间网络阻止了该端口。

### 5. 虚拟机环境

如果 ROS 和 Streamlit 运行在 VMware、VirtualBox 等虚拟机中：

- **桥接网络（Bridged）最容易访问**。改为桥接后，用 `hostname -I`
  重新查询虚拟机 IP。
- 如果使用 **NAT**，宿主机的 `8501` 和 `8502` 端口需要转发到虚拟机的
  同一端口，然后客户端访问宿主机的局域网 IP。
- 虚拟机内部地址（例如 `192.168.91.x`）通常不能被局域网其他电脑直接访问。

### 6. 不在同一局域网：使用 SSH 端口转发

如果客户端能通过 SSH 登录服务器，但不方便直接访问服务器的 `8501`、
`8502` 端口，可以在客户端执行：

```bash
ssh -L 8501:127.0.0.1:8501 \
    -L 8502:127.0.0.1:8502 \
    用户名@服务器IP
```

保持 SSH 连接，然后在客户端浏览器打开：

```text
http://localhost:8501
```

这样不需要把 Streamlit 或视频流直接暴露到公网。

## 五、网页中的相机和识别模式

### 相机来源

| 页面选项 | 实际使用的相机 | 说明 |
| --- | --- | --- |
| ROS 话题 | 服务器端 D435i | 推荐，配合 `realsense2_camera` |
| RealSense 直连 | 服务器端 D435i | 不经过 ROS，需要 `pyrealsense2` |
| 浏览器拍照 | 客户端浏览器所在电脑的摄像头 | 从其他电脑访问时，使用的是那台电脑的摄像头 |

因此，从另一台电脑访问网页时，如果要识别服务器上的 D435i 画面，应使用
`ROS 话题` 或 `RealSense 直连`，不要选择 `浏览器拍照`。

### 识别模式

| 模式 | 适用场景 |
| --- | --- |
| `⚡ 快速` | 单帧直接检测和比对，延迟低 |
| `🎯 完整` | YOLO + 质量过滤 + 多帧投票，结果更稳定，但需要多帧才确认身份 |

完整模式会同时加载 InsightFace 和 YOLO。不要同时运行 ROS 的
`face_recognition.launch` 和网页完整模式，否则在内存较小的机器上容易 OOM。

## 六、常见问题

### 网页打不开

确认 Streamlit 仍在运行：

```bash
ss -ltnp | grep -E ':8501|:8502'
```

应该看到类似：

```text
0.0.0.0:8501
0.0.0.0:8502
```

如果只看到 `127.0.0.1:8501`，请使用启动脚本，或显式添加：

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 页面能打开，但实时视频没有画面

依次检查：

1. 是否选择了正确的取流方式和话题。
2. 服务器端相机是否已经启动。
3. 客户端是否能访问 `8502` 端口。
4. 页面里的 `视频流端口` 是否与防火墙、SSH 转发或虚拟机转发一致。
5. 浏览器开发者工具的网络面板中，`/stream` 和 `/status` 请求是否失败。

检查 ROS 图像：

```bash
rostopic list | grep image
rostopic hz /camera/color/image_raw
```

### 提示 ROS master 未运行

自动启动相机时，`roslaunch` 会自动启动 ROS master。也可以先手动启动：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

然后在另一个终端启动相机或网页。

自动启动的日志位于：

```text
/tmp/face_webapp_logs/camera.log
/tmp/face_webapp_logs/face_node.log
```

网页侧边栏的 `🚀 ROS 启动（roslaunch）` 面板中也可以直接查看日志。

### 相机没有图像

- 检查 D435i 是否插在 USB 3.0 接口上。
- 检查是否已有其他进程打开了相机。
- 检查彩色话题名称是否为 `/camera/color/image_raw`。
- 先关闭网页和 ROS，再重新启动相机驱动。
- 如果使用直连模式，确认 `pyrealsense2` 已安装且 ROS 相机驱动已停止。

### 浏览器拍照提示没有权限

浏览器通常只允许 `localhost` 或 HTTPS 页面调用客户端摄像头。从其他电脑
使用 `http://服务器IP:8501` 访问时，`浏览器拍照` 可能被拦截。

推荐不经过浏览器，直接使用服务器端 D435i 抓拍。也可以通过 SSH 隧道在
客户端打开 `http://localhost:8501`，或为 Streamlit 配置 HTTPS。

### 端口已被占用

查看占用进程：

```bash
ss -ltnp | grep 8501
```

换一个网页端口：

```bash
./start_webapp.sh 8601 1
```

然后访问 `http://服务器IP:8601`，并相应放行 `8601`。

### 关闭程序

- 在启动 Streamlit 的终端按 `Ctrl+C` 停止网页。
- 在 ROS 节点终端按 `Ctrl+C` 停止 `roslaunch`。
- 如果相机由网页自动启动，先在网页侧边栏点击 `⏹ 停止`，再关闭网页。
- 停止实时识别时，先点击 `⏹ 停止视频`，释放相机和视频流端口。

## 七、安全说明

当前网页没有登录认证和权限控制。只建议在可信局域网内使用，不要直接把
`8501`、`8502` 映射到公网。

需要外网访问时，优先使用 VPN 或 SSH 隧道；如果必须暴露服务，请在反向
代理层增加 HTTPS、身份认证和访问控制。
