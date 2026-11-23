# 人脸识别网页版（Streamlit）

把 `face_recognition_ros` 的人脸识别能力做成网页，支持**人脸录入、查看、添加、删除**，
并与 ROS 节点共用同一套照片目录和特征数据库。

## 目录结构

```
webapp/
├── app.py            # Streamlit 界面（入口）
├── face_core.py      # 核心逻辑：模型、特征提取、数据库增删改、识别
├── recognizer.py     # 与 ROS 节点一致的实时识别流水线（YOLO+质量过滤+投票）
├── camera.py         # D435i 取流（ROS 话题 / pyrealsense2 直连）
├── live_stream.py    # MJPEG 实时视频流（采集线程 + 识别线程解耦）
├── config.py         # 路径与参数配置（支持环境变量覆盖）
├── requirements.txt  # 依赖
└── .streamlit/config.toml
```

## 数据存放位置（与 ROS 节点一致）

| 内容 | 路径 |
| --- | --- |
| 人脸照片 | `face_recognition_ros/face_data/<姓名>/*.jpg` |
| 特征数据库 | `face_recognition_ros/model/chenzhen_face_database.pkl` |
| 陌生人截图 | `face_recognition_ros/stranger_faces/stranger_*.jpg` |
| 识别画面存档 | `face_recognition_ros/recognition_snapshots/<known/stranger/mixed>/*.jpg` |

数据库格式为 `{姓名: {"embeddings": [...], "template": [...], "images": {...}}}`，
与 `scripts/face_recognition_node.py` 的读取逻辑完全兼容。

## 安装

系统已安装 `insightface / onnxruntime / opencv / numpy`（ROS Noetic 环境）。
只需补装 Streamlit：

```bash
pip3 install -r requirements.txt
# 或者
pip3 install "streamlit==1.36.0"
```

## 启动

### 同一局域网内的其他电脑访问

先确认两台电脑连接的是同一个局域网（同一路由器 / Wi-Fi），然后在运行网页的电脑上执行：

```bash
cd /home/xiangsi/catkin_ws/src/face_recognition_ros/webapp
./start_webapp.sh 8501 0
```

终端会显示两行地址，例如：

```text
本机访问:   http://localhost:8501
其他电脑:   http://192.168.1.23:8501
```

在另一台电脑的浏览器中打开第二行地址即可。也可以在运行网页的电脑上单独查看 IP：

```bash
hostname -I
```

输出中通常形如 `192.168.x.x` 或 `10.x.x.x` 的地址就是局域网 IP。不要使用 `127.0.0.1`
或 `localhost`，它们只代表电脑自己。

如果项目运行在 VMware、VirtualBox 等虚拟机中，而虚拟机使用 **NAT** 网络，`hostname -I`
显示的可能是只有宿主机和虚拟机之间才能访问的地址（例如 `192.168.91.x`），同局域网内的
其他电脑无法直接打开。将虚拟机网卡改为 **桥接（Bridged）** 后重新查看 IP；或在虚拟机软件里
把宿主机的 `8501`、`8502` 端口转发到虚拟机的同一端口，然后让其他电脑访问宿主机的局域网 IP。

项目已通过 `.streamlit/config.toml` 固定设置 `address = "0.0.0.0"` 和 `port = 8501`，
因此直接执行 `streamlit run app.py` 也会监听所有网卡。启动脚本还会显式传入相同的参数。

如果另一台电脑仍打不开，先在另一台电脑上测试端口（Windows PowerShell）：

```powershell
Test-NetConnection 192.168.1.23 -Port 8501
```

`TcpTestSucceeded` 应为 `True`。若为 `False`，检查运行网页电脑的防火墙并放行端口：

```bash
sudo ufw status
sudo ufw allow 8501/tcp
sudo ufw allow 8502/tcp
```

最后一条 `8502` 用于「📹 实时识别」页面的 MJPEG 视频流；不使用该页面时可以不开放。

> 该网页目前没有登录认证。只建议在可信局域网内开放，不要把 `8501` 端口直接映射到公网。

> 实时视频流会额外监听 **8502** 端口（MJPEG），用「📹 实时识别」页时需要放行/转发它。

## 打开网页就自动运行 roslaunch

不用再单开终端跑 `roslaunch`：页面打开时会自动把 ROS 节点拉起来（后台子进程），
侧边栏「🚀 ROS 启动（roslaunch）」面板里可以随时启动 / 停止 / 看日志。

```bash
cd /home/xiangsi/catkin_ws/src/face_recognition_ros/webapp
./start_webapp.sh                 # 自动 source ROS 环境 + 打开网页自动启动相机驱动
./start_webapp.sh 8501 0          # 第 2 个参数 0 -> 关闭自动启动，只开网页
```

行为说明：

- **幂等**：同一目标只启动一次。若话题 `/camera/color/image_raw` 上已经有发布者
  （说明你在别处跑过 `roslaunch`），网页会直接复用，不会重复启动。
- **停止**：面板里的「⏹ 停止」会对整个进程组发 SIGINT，`roslaunch` 会按正常流程
  关掉所有子节点（只停网页启动的进程，不动你手动跑的）。
- **日志**：每个目标的输出落在 `/tmp/face_webapp_logs/<目标>.log`，面板里可直接看尾部，
  启动失败（比如相机没插）会直接显示出来。
- **识别由谁做**：默认只启动相机驱动，识别在网页里跑。若要连 ROS 识别节点一起启动，
  把启动目标切成 `face_node`（注意：它自带 YOLO，约 2GB 内存，
  不要和网页「🎯 完整模式」同时使用，本机 3.8GB 容易 OOM）。

想改默认行为，用环境变量（或直接改 `config.py`）：

```bash
# 默认只起相机驱动；改成起完整识别节点
FACE_WEBAPP_LAUNCH_TARGET=face_node ./start_webapp.sh

# 起任意命令（例如 usb_cam 版）
FACE_WEBAPP_LAUNCH_CMD="roslaunch face_recognition_ros face_recognition_lat.launch" \
FACE_WEBAPP_LAUNCH_TOPIC=/usb_cam/image_raw ./start_webapp.sh

# 完全关闭自动启动
FACE_WEBAPP_AUTOLAUNCH=0 ./start_webapp.sh
```

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `FACE_WEBAPP_AUTOLAUNCH` | 打开网页是否自动启动 | `1` |
| `FACE_WEBAPP_LAUNCH_TARGET` | 启动目标 `camera` / `face_node` / `custom` | `camera` |
| `FACE_WEBAPP_LAUNCH_CMD` | 自定义命令（非空时下拉框多出 `custom`） | 空 |
| `FACE_WEBAPP_LAUNCH_TOPIC` | 自定义目标的就绪话题（用来判断启动成功） | `/camera/color/image_raw` |
| `FACE_WEBAPP_LAUNCH_LOG_DIR` | 启动日志目录 | `/tmp/face_webapp_logs` |
| `FACE_WEBAPP_ROS_SETUP` | ROS 环境脚本 | `/opt/ros/noetic/setup.bash` |
| `FACE_WEBAPP_WS_SETUP` | 工作空间 setup.bash | `~/catkin_ws/devel/setup.bash` |

> 想让「开机 / 登录桌面」就自动跑起来，可以把这个脚本挂到
> `~/.config/autostart/*.desktop`（图形登录）或写成 systemd user 服务，
> 脚本内部已经做了 `source` 和 `cd`，直接指向 `start_webapp.sh` 即可。

## 功能说明

- **📊 概览**：人数、照片数、特征总数与成员缩略图。
- **➕ 人脸录入**：填写姓名，上传图片或直接拍照，自动检测人脸并提取 512 维特征；
  支持「新建身份」或「追加到已有身份」，可设置最小检测分过滤模糊/侧脸。
  三种来源：**📁 上传图片** / **📹 D435i 抓拍**（推荐，不依赖浏览器权限）/
  **📷 浏览器拍照**。

  「📹 D435i 抓拍」支持 **🔴 开启预览** → 实时看到相机画面 → **📸 抓拍一张**
  （从当前预览画面取帧，可连拍多张）→ **开始录入**。预览为纯取流模式，
  不跑识别，CPU 占用很低。

  预览画面的**显示尺寸**可在「预览尺寸」里切换，四种模式都保证整帧可见（不裁切）：

  | 选项 | 效果 |
  | --- | --- |
  | 自适应宽度（推荐） | 缩放到容器宽度内，长宽比不变，整帧完整显示 |
  | 大（高 720px） | 同上，但画面更大，适合看清远处人脸 |
  | 原始尺寸 1:1（640×480） | 按相机像素直接显示，不做放大 |
  | 全宽 | 铺满页面宽度，很高，必要时页面内滚动 |

  「📹 实时识别」页的「视频流参数」里也有同一个开关。
- **🗂️ 人脸库管理**：查看每个身份的全部照片；支持**删除单张照片**、**删除整个身份**、
  **重命名**、**按照片重建特征**，以及**重建整个数据库**。

  照片以**统一的正方形缩略图网格**展示：等比缩放 + 居中留边（不裁切、不变形），
  所以手机竖图、相机横图混在一起也不会有高矮不齐、撑破列的情况；每行严格对齐。
  网格上有两个开关：**每行张数**（2~6）和**缩略图大小**（小/中/大），
  每张缩略图下面标着文件名和原始分辨率，右键「另存为」可以看原图。
  概览页的成员卡片、「➕ 人脸录入」的录入结果与抓拍列表用的是同一套网格。
- **🔍 识别测试**：上传图片，实时与数据库比对并显示相似度（不影响数据库）。
- **📹 实时识别（D435i）**：从 Intel RealSense D435i 取流并做实时人脸识别，见下节。
  判为陌生人的脸会显示**红框**并保存裁图；同一次任务中同一张脸只保存一次。
  含人的完整识别画面也会存档，并按**数据库人员 / 陌生人 / 混合**分类。

## 实时识别：连接 D435i 深度相机

网页端支持两种取流方式，「📹 实时识别」页面顶部有 **🔌 连接检查** 面板可随时诊断。

### 路线 A（推荐）：ROS 话题 + realsense2_camera

复用你项目里已有的相机驱动，网页通过 `rospy` 订阅图像话题。**无需额外安装**。

```bash
# 终端 1：启动相机驱动（align_depth 会让深度对齐到彩色）
source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true

# 终端 2：启动网页（同样需要 source ROS 环境，保证能 import rospy）
source /opt/ros/noetic/setup.bash
cd /home/xiangsi/catkin_ws/src/face_recognition_ros/webapp
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

页面里选择「ROS 话题」，默认话题：

| 用途 | 话题 |
| --- | --- |
| 彩色 | `/camera/color/image_raw` |
| 对齐深度 | `/camera/aligned_depth_to_color/image_raw` |

### 路线 B：pyrealsense2 直连（无需 ROS）

适合不启动 ROS、单独部署网页的场景：

```bash
pip3 install pyrealsense2
```

页面里选择「RealSense 直连」，可选分辨率 / 帧率 / 是否启用深度。

> ⚠️ **同一个相机不能被两个进程同时打开**：用路线 B 时请先关掉 `realsense2_camera`，
> 反之亦然。同一时刻只能二选一。

### 页面操作

- **▶️ 开始实时视频**：在页面里播放**连续视频流**（MJPEG），人脸框实时叠加。
- **⏹ 停止视频**：停止取流并释放相机。
- **📸 抓拍一帧**：抓取单张画面识别一次。
- **♻️ 重置投票**：清空多帧跟踪与投票历史（完整模式）。
- **🕵️ 本次任务陌生人截图**：只查看当前任务保存的陌生人脸，运行中自动刷新。
  每个不同陌生人默认只保存 **1 张**，不会因为视频连续帧重复写盘。
- **🖼️ 本次任务结果**：左侧独立页面，展示当前任务保存的图片和逐人脸识别结果。

### 实时视频流是怎么做到"流畅"的

Streamlit 本身是"一帧一帧刷新"的模型，直接往页面塞图片必然卡顿。
所以这里用 **MJPEG over HTTP** 推流，把两件事解耦：

```
采集线程  ──> 按相机帧率抓帧 + 光流移动人脸框 + JPEG ──> /stream
预检测线程 ──> SCRFD 320×320 快速检测人脸框（约 4 次/秒）
识别线程  ──> 按 CPU 速度（约 0.6～1.2 s/次）更新姓名/陌生人结果
```

- 画面是**连续视频**（约 15 fps），人脸标注约每秒更新一到两次。
- 快速检测负责让人脸框**约 0.1～0.3 秒出现**，并在两次正式识别之间用光流让框跟随
  人脸移动；身份和陌生人结果仍按正式识别速度更新。
- 状态通过 `/status` 每秒轮询显示：画面帧率、快速框帧率、识别耗时、人脸数、播放端数量。
- 视频流端口默认 **8502**（可在页面上改），用 `http://<浏览器所在主机的地址>:8502/stream` 访问。
  页面会自动用浏览器当前的 hostname 拼地址，所以本机、局域网访问都能直接播。

> 如果是从别的机器访问、或走 SSH 端口转发，记得**把 8502 也转发/放行**，
> 例如 `ssh -L 8501:localhost:8501 -L 8502:localhost:8502 用户@主机`。

### 两种识别模式

| 模式 | 流程 | 适用 |
| --- | --- | --- |
| ⚡ 快速 | 整帧直接 InsightFace 检测 + 识别，**不投票** | 单帧即可出结果，延迟最低 |
| 🎯 完整 | **与 `face_recognition.launch` 的识别节点完全一致**：YOLO 人体检测 → 人体区域内人脸检测/特征 → 质量过滤 → 多帧跟踪 + 投票 | 结果更稳、抗抖动 |

完整模式的参数默认直接读取 `config/params.yaml`（与节点同源），可在页面上临时调整：
阈值、YOLO 开关、质量过滤、投票窗口、最少票数、计票最低相似度、跟踪最大位移等。

> 完整模式需要累积票数（默认 ≥8 票）才判定身份，所以单次抓拍通常显示为"识别中…"，
> 连续监测或多次抓拍后会给出确定结果。

### 陌生人红框与自动保存

当最高相似度低于识别阈值时，人脸会显示为**红框**，并保存一张带少量边缘上下文的人脸裁图。
快速模式得到陌生结果时保存；完整模式在多帧投票确认后才保存。
截图使用人脸特征去重，同一次任务中同一个陌生人只保存一次。
不同陌生人会分别保存，默认不限制总人数。若确实需要限制，可通过
`FACE_WEBAPP_STRANGER_MAX_PER_TASK` 设置当前任务的截图硬上限；`0` 表示不限。

陌生人截图独立存放在 `stranger_faces/`，**不会写入人脸数据库**，因此后续不会因为截图文件
本身被误识别为已知成员。实时识别页的「🕵️ 本次任务陌生人截图」只展示当前任务，
运行期间新截图会自动出现；之前的任务截图仍保留在磁盘，但不会混入当前任务。

### 识别画面存档（数据库人员 / 陌生人）

只有当一张人脸得到确认结果后，系统才会保存一次**完整标注画面**，不会随着视频帧
持续重复保存。数据库人员使用绿框和姓名，陌生人使用红框，存档图片底部还会增加
类型条，明确标出：

- `known/`：数据库人员，类型条显示姓名。
- `stranger/`：陌生人，类型条显示陌生人数量。
- `mixed/`：同一画面同时有数据库人员和陌生人。

同一种识别结果在**当前任务中只保存一次**。每次开始实时视频或抓拍一帧都会建立新任务；
左侧「🖼️ 本次任务结果」会展示当前任务保存的图片、数据库姓名或陌生人数量及相似度。
实时识别页中的「🖼️ 本次任务识别画面存档」也只展示当前任务，不包含之前任务的图片。

实时视频运行期间，「🖼️ 本次任务结果」会通过视频流端口自动刷新；新结果保存后约 1 秒
内就会出现，不需要手动重载页面。停止视频后，该页面仍可静态查看任务已经保存的全部结果。

### 排查

- 点「重新检查」看 ROS master 是否在线、`pyrealsense2` 是否安装、有哪些彩色话题。
- 提示「未收到图像数据」：确认话题名正确、相机已启动；`rostopic list` 里应能看到该话题。
- 若想获得真正低延迟的连续视频，可改用 `streamlit-webrtc`（本仓库未默认启用）。

### ⚠️ 性能与内存提醒（本机实测）

- 本机是 **2 核 CPU、无 NVIDIA GPU**，`onnxruntime` 只有 `CPUExecutionProvider`，
  因此识别在 CPU 上运行，640×480 下约 **0.6～1.0 秒/次**（实测）：

  | 画面帧率上限 | 实际画面 | 识别耗时 |
  | --- | --- | --- |
  | 25 fps | 22.7 fps | 1.02 s/次 |
  | **15 fps（默认）** | **14.9 fps** | **0.61 s/次** |
  | 10 fps | 9.9 fps | 0.60 s/次 |
  | 5 fps | 5.1 fps | 0.72 s/次 |

  画面和识别抢 CPU，所以不是越高越好；默认 15 fps 兼顾流畅与识别速度。
  如果加装了 GPU 版，可设
  `FACE_WEBAPP_PROVIDER=CUDAExecutionProvider,CPUExecutionProvider` 大幅提速。
- 内存只有 **3.8 GB**：InsightFace 单个实例约 **1 GB**，再加 YOLO 约 **1.24 GB**。
  **不要同时运行 ROS 识别节点和网页的「🎯 完整模式」**，两者都要加载 YOLO + InsightFace，
  容易触发 OOM。需要同时用的话，节点和网页之间选一个跑，或网页改用「⚡ 快速模式」。

> 环境说明：本机 `cv_bridge` 的 `cv2_to_imgmsg` 存在兼容问题（发布方向），
> 因此 `camera.py` 只使用 `imgmsg_to_cv2`（订阅方向），实测正常。

## 浏览器摄像头打不开（"This app would like to use your camera"）

这不是代码问题：**浏览器只允许 HTTPS 或 localhost 页面调用摄像头**。
如果你在另一台电脑上用 `http://<IP>:8501` 打开，权限会被浏览器拦截。

可选方案（任选其一）：

1. **改用 D435i 抓拍**（最省事）：录入页的「📹 D435i 抓拍」直接从深度相机取图，
   完全不经过浏览器摄像头。
2. **用 localhost**：在服务器本机的浏览器打开 `http://localhost:8501`。
3. **SSH 端口转发**（浏览器在另一台电脑上）：

   ```bash
   ssh -L 8501:localhost:8501 -L 8502:localhost:8502 用户名@<服务器IP>
   ```

   然后浏览器访问 `http://localhost:8501`。
4. **给 Streamlit 配自签名 HTTPS**：

   ```bash
   mkdir -p ~/.streamlit/certs && cd ~/.streamlit/certs
   openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
     -keyout key.pem -out cert.pem -subj "/CN=localhost"
   streamlit run app.py \
     --server.sslCertFile=$HOME/.streamlit/certs/cert.pem \
     --server.sslKeyFile=$HOME/.streamlit/certs/key.pem
   ```

   访问 `https://<IP>:8501` 并选择"继续前往"。
5. **Chrome 白名单**：打开 `chrome://flags/#unsafely-treat-insecure-origin-as-secure`，
   把 `http://<IP>:8501` 加进去并重启浏览器。

另外检查：地址栏左侧权限图标是否把摄像头设成了"阻止"；系统隐私设置里是否允许浏览器用摄像头。

## 与 ROS 节点的关系

- 网页端的增删改会**立即写入** `face_data/` 和特征数据库文件。
- ROS 节点在启动时加载数据库，因此网页修改后**重启节点**（`roslaunch` 重新运行）
  即可生效；运行中的节点不会热更新。
- 识别算法一致：查询特征 L2 归一化后与模板/逐条特征做余弦相似度，
  阈值默认 `0.4`（可在 `识别测试` 页临时调整，正式阈值见 `config/params.yaml`）。

## 可选：环境变量

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `FACE_WEBAPP_DATA_DIR` | 照片目录 | `<包>/face_data` |
| `FACE_WEBAPP_DB` | 特征数据库 | `<包>/model/chenzhen_face_database.pkl` |
| `FACE_WEBAPP_STRANGER_DIR` | 陌生人截图目录 | `<包>/stranger_faces` |
| `FACE_WEBAPP_STRANGER_SIMILARITY` | 截图去重的人脸相似度阈值 | `0.55` |
| `FACE_WEBAPP_STRANGER_MAX_PER_TASK` | 每个任务陌生截图硬上限，`0` 为不限 | `0` |
| `FACE_WEBAPP_STRANGER_CROP_MARGIN` | 人脸框向外扩展比例 | `0.20` |
| `FACE_WEBAPP_RECOGNITION_SNAPSHOT_DIR` | 识别画面存档根目录 | `<包>/recognition_snapshots` |
| `FACE_WEBAPP_RECOGNITION_SNAPSHOT_FONT` | 存档类型条中文字体 | 文泉驿正黑 |
| `FACE_WEBAPP_THRESHOLD` | 识别阈值 | `0.4` |
| `FACE_WEBAPP_MODEL` | 模型名 | `buffalo_l` |
| `FACE_WEBAPP_ROOT` | 模型根目录 | `~/.insightface` |
| `FACE_WEBAPP_PROVIDER` | onnxruntime 后端 | `CPUExecutionProvider` |

> 有 GPU 且安装了 `onnxruntime-gpu` 时，可设
> `FACE_WEBAPP_PROVIDER=CUDAExecutionProvider,CPUExecutionProvider` 加速。
