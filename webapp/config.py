#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别网页应用 —— 路径与参数配置。

所有路径都可以通过环境变量覆盖，方便在不同机器上部署：
    FACE_WEBAPP_DATA_DIR   人脸照片目录      (默认 <pkg>/face_data)
    FACE_WEBAPP_DB         人脸特征数据库    (默认 <pkg>/model/chenzhen_face_database.pkl)
    FACE_WEBAPP_STRANGER_DIR 陌生人截图目录  (默认 <pkg>/stranger_faces)
    FACE_WEBAPP_STRANGER_SIMILARITY 陌生人截图去重相似度 (默认 0.55)
    FACE_WEBAPP_STRANGER_MAX_PER_TASK 每个任务最多保存张数
                                      (默认 0，表示不限；每个陌生人仍只存一张)
    FACE_WEBAPP_RECOGNITION_SNAPSHOT_DIR 识别画面存档目录
                                          (默认 <pkg>/recognition_snapshots)
    FACE_WEBAPP_THRESHOLD  识别相似度阈值    (默认 0.4)
    FACE_WEBAPP_MODEL      模型名称          (默认 buffalo_l)
    FACE_WEBAPP_ROOT       模型根目录        (默认 ~/.insightface)
    FACE_WEBAPP_PROVIDER   onnxruntime 执行后端, 逗号分隔 (默认 CPUExecutionProvider)

自动启动 roslaunch（打开网页即运行）：
    FACE_WEBAPP_AUTOLAUNCH       打开网页时是否自动启动, 1/0 (默认 1)
    FACE_WEBAPP_LAUNCH_TARGET    启动目标 camera / face_node / custom (默认 camera)
    FACE_WEBAPP_LAUNCH_CMD       自定义命令（字符串, 简单按空格拆分）
    FACE_WEBAPP_LAUNCH_TOPIC     自定义目标的就绪话题（用于等待）
    FACE_WEBAPP_LAUNCH_LOG_DIR   日志目录 (默认 /tmp/face_webapp_logs)
    FACE_WEBAPP_ROS_SETUP        ROS 环境脚本 (默认 /opt/ros/noetic/setup.bash)
    FACE_WEBAPP_WS_SETUP         工作空间 setup.bash (默认 ~/catkin_ws/devel/setup.bash)
"""

import os

# 包根目录: .../face_recognition_ros
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 人脸照片目录: face_data/<姓名>/*.jpg
FACE_DATA_DIR = os.environ.get(
    "FACE_WEBAPP_DATA_DIR", os.path.join(PKG_DIR, "face_data")
)

# ROS 节点参数文件（实时监测的默认参数直接读它，保证与节点一致）
PARAMS_FILE = os.environ.get(
    "FACE_WEBAPP_PARAMS", os.path.join(PKG_DIR, "config", "params.yaml")
)

# 人脸特征数据库文件
# 默认复用 ROS 节点 params.yaml 中指向的同一个文件，
# 这样网页里录入/删除的人脸与 ROS 节点识别保持一致。
DB_FILE = os.environ.get(
    "FACE_WEBAPP_DB", os.path.join(PKG_DIR, "model", "chenzhen_face_database.pkl")
)

# 识别为陌生人时保存的人脸截图。该目录不参与人脸库/特征库，避免陌生人
# 被下一次识别误判为已知身份。
STRANGER_DIR = os.environ.get(
    "FACE_WEBAPP_STRANGER_DIR", os.path.join(PKG_DIR, "stranger_faces")
)

# 同一个陌生人只保存一次。使用人脸特征余弦相似度去重，快速模式没有
# 跟踪 ID 时也能避免每个识别帧都写一张图。
STRANGER_SIMILARITY = float(os.environ.get(
    "FACE_WEBAPP_STRANGER_SIMILARITY", "0.55"
))
STRANGER_MAX_PER_TASK = max(0, int(os.environ.get(
    "FACE_WEBAPP_STRANGER_MAX_PER_TASK", "0"
)))
STRANGER_CROP_MARGIN = float(os.environ.get(
    "FACE_WEBAPP_STRANGER_CROP_MARGIN", "0.20"
))

# 保存带完整标注的识别画面，按 known / stranger / mixed 分类存放。
RECOGNITION_SNAPSHOT_DIR = os.environ.get(
    "FACE_WEBAPP_RECOGNITION_SNAPSHOT_DIR",
    os.path.join(PKG_DIR, "recognition_snapshots"),
)
RECOGNITION_SNAPSHOT_FONT_PATH = os.environ.get(
    "FACE_WEBAPP_RECOGNITION_SNAPSHOT_FONT",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

# 识别阈值（与 ROS 节点 config/params.yaml 的 threshold 保持一致）
THRESHOLD = float(os.environ.get("FACE_WEBAPP_THRESHOLD", "0.4"))

# InsightFace 模型
MODEL_NAME = os.environ.get("FACE_WEBAPP_MODEL", "buffalo_l")
MODEL_ROOT = os.environ.get("FACE_WEBAPP_ROOT", os.path.expanduser("~/.insightface"))
PROVIDERS = [
    p.strip()
    for p in os.environ.get(
        "FACE_WEBAPP_PROVIDER", "CPUExecutionProvider"
    ).split(",")
    if p.strip()
]

# 默认人脸检测分辨率
DET_SIZE = (640, 640)

# 允许上传的图片扩展名
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ============================================================
# roslaunch 自动启动（由 launch_manager.py 使用）
# ============================================================


def _as_bool(text, default=True):
    if text is None:
        return default
    return str(text).strip().lower() not in ("0", "false", "no", "off", "")


def _split_cmd(text):
    return [p for p in str(text).split() if p]


# 打开网页时是否自动启动 roslaunch
AUTOLAUNCH = _as_bool(os.environ.get("FACE_WEBAPP_AUTOLAUNCH"), True)

# 自动启动的目标：camera / face_node / custom
LAUNCH_TARGET = os.environ.get("FACE_WEBAPP_LAUNCH_TARGET", "camera").strip()

# 自定义目标（FACE_WEBAPP_LAUNCH_CMD 非空时会出现在页面下拉框里）
LAUNCH_CMD = _split_cmd(os.environ.get("FACE_WEBAPP_LAUNCH_CMD", ""))
LAUNCH_TOPIC = os.environ.get("FACE_WEBAPP_LAUNCH_TOPIC",
                              "/camera/color/image_raw").strip()

# 每个目标的日志文件目录
LAUNCH_LOG_DIR = os.environ.get("FACE_WEBAPP_LAUNCH_LOG_DIR",
                                "/tmp/face_webapp_logs")

# 启动 roslaunch 时先 source 这两个环境脚本
ROS_SETUP = os.environ.get("FACE_WEBAPP_ROS_SETUP", "/opt/ros/noetic/setup.bash")
WS_SETUP = os.environ.get(
    "FACE_WEBAPP_WS_SETUP",
    os.path.join(os.path.expanduser("~"), "catkin_ws", "devel", "setup.bash"),
)
