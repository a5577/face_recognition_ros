#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度相机（Intel RealSense D435i）采集模块。

提供两种取流方式，接口一致（read / close / info）：

1. RosFrameSource   —— 通过 ROS 话题订阅（推荐，配合 realsense2_camera）
       roscore 已运行时：roslaunch realsense2_camera rs_camera.launch align_depth:=true
       彩色: /camera/color/image_raw
       对齐深度: /camera/aligned_depth_to_color/image_raw

2. RealSenseSource  —— 通过 pyrealsense2 直接打开相机（无需 ROS）
       需要: pip3 install pyrealsense2

统一返回 BGR 彩色图与（可选的）伪彩色深度图。
"""

import threading
import time

import cv2
import numpy as np


# ============================================================
# 深度图伪彩色
# ============================================================

def depth_to_colormap(depth, max_range_m=4.0, colormap=cv2.COLORMAP_JET):
    """
    把深度图转成便于观看的伪彩色图。

    支持 uint16（毫米）与 float32（米）。无效值(0/NaN)显示为黑色。
    """
    if depth is None:
        return None
    d = np.asarray(depth)
    if d.dtype == np.uint16:
        d = d.astype(np.float32) / 1000.0          # mm -> m
    else:
        d = d.astype(np.float32)                   # 假设已是米

    valid = np.isfinite(d) & (d > 0.05)
    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    d = np.clip(d, 0.0, max_range_m)
    norm = np.zeros_like(d, dtype=np.uint8)
    norm[valid] = np.clip(d[valid] / max_range_m * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    colored[~valid] = (0, 0, 0)
    return colored


# ============================================================
# 取流基类
# ============================================================

class FrameSource(object):
    """统一的取流接口。"""

    name = "source"

    def read(self, timeout=1.0):
        """
        返回 (color_bgr, depth_vis, stamp)。
        超时/无数据时返回 (None, None, None)。
        """
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================
# 路线 A：ROS 话题
# ============================================================

def ros_core_ready():
    """判断 ROS master 是否可达。"""
    try:
        import rosgraph
        return rosgraph.is_master_online()
    except Exception:
        return False


def list_ros_topics():
    """列出当前 ROS 话题（用于界面诊断）。"""
    try:
        import rosgraph
        # getSystemState() 返回 [publishers, subscribers, services]，
        # 其中每项是 [[话题名, [节点列表]], ...]，需要取出话题名。
        state = rosgraph.Master("/").getSystemState()
        return sorted({entry[0] for entry in state[0]})
    except Exception:
        return []


def ensure_rospy(node_name="face_web_live"):
    """在当前进程初始化 rospy（幂等）。"""
    import rospy
    _drop_rospy_atexit()
    if not rospy.core.is_initialized():
        # disable_signals: Streamlit 主线程不允许 rospy 注册信号处理
        rospy.init_node(node_name, anonymous=True, disable_signals=True)
    return rospy


def _drop_rospy_atexit():
    """
    去掉 rospy 的 atexit 钩子。

    rospy 在 import 时就会注册 `rospy.core._ros_atexit`，它会在解释器退出时
    join 内部线程。此时 Streamlit 的 asyncio 事件循环通常已经关闭，
    若再有 SIGINT 送达就会触发 Streamlit 的信号处理，
    抛 "RuntimeError: Event loop is closed"。

    本应用只是订阅方，不依赖该清理，直接注销即可。
    """
    try:
        import atexit
        import rospy.core
        atexit.unregister(rospy.core._ros_atexit)
    except Exception:
        pass


class RosFrameSource(FrameSource):
    name = "ROS 话题"

    def __init__(self, color_topic="/camera/color/image_raw",
                 depth_topic=None, depth_max_range=4.0):
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        self.rospy = ensure_rospy()
        self.bridge = CvBridge()
        self.color_topic = color_topic
        self.depth_topic = depth_topic or None
        self.depth_max_range = depth_max_range

        self._lock = threading.Lock()
        self._color = None
        self._depth = None
        self._color_seq = 0
        self._depth_seq = 0
        self._subs = []

        self._subs.append(self.rospy.Subscriber(
            color_topic, Image, self._on_color,
            queue_size=1, buff_size=2 ** 24))
        if self.depth_topic:
            self._subs.append(self.rospy.Subscriber(
                self.depth_topic, Image, self._on_depth,
                queue_size=1, buff_size=2 ** 24))

    def _on_color(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self._lock:
            self._color = img
            self._color_seq += 1

    def _on_depth(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return
        vis = depth_to_colormap(raw, self.depth_max_range)
        with self._lock:
            self._depth = vis
            self._depth_seq += 1

    def read(self, timeout=1.0):
        deadline = time.time() + timeout
        with self._lock:
            last = self._color_seq
        # 1) 等待一帧新的彩色图
        while time.time() < deadline:
            with self._lock:
                if self._color is not None and self._color_seq != last:
                    break
            time.sleep(0.01)
        # 2) 若启用了深度，尽量等到深度就绪（首帧深度可能稍晚于彩色到达）
        while self.depth_topic and time.time() < deadline:
            with self._lock:
                if self._depth is not None:
                    break
            time.sleep(0.01)
        with self._lock:                      # 超时则返回当前最新帧
            if self._color is None:
                return None, None, None
            return (self._color.copy(),
                    None if self._depth is None else self._depth.copy(),
                    time.time())

    def close(self):
        for s in self._subs:
            try:
                s.unregister()
            except Exception:
                pass
        self._subs = []


# ============================================================
# 路线 B：pyrealsense2 直连
# ============================================================

def pyrealsense_available():
    try:
        import pyrealsense2  # noqa: F401
        return True
    except Exception:
        return False


class RealSenseSource(FrameSource):
    name = "RealSense 直连"

    def __init__(self, width=640, height=480, fps=30,
                 enable_depth=True, depth_max_range=4.0):
        import pyrealsense2 as rs

        self.rs = rs
        self.depth_max_range = depth_max_range

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.enable_depth = enable_depth
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color) if enable_depth else None

        # 读取一次内参/设备信息（可选）
        try:
            dev = profile.get_device()
            self.device_name = dev.get_info(rs.camera_info.name)
            self.serial = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            self.device_name = "RealSense"
            self.serial = ""

    def read(self, timeout=1.0):
        try:
            frames = self.pipeline.wait_for_frames(
                timeout_ms=int(max(timeout, 0.05) * 1000))
        except Exception:
            return None, None, None

        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None, None
        color = np.asanyarray(color_frame.get_data())   # BGR

        depth_vis = None
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data())  # uint16 mm
                depth_vis = depth_to_colormap(depth, self.depth_max_range)
        return color, depth_vis, time.time()

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ============================================================
# 工厂
# ============================================================

def create_source(mode, color_topic="/camera/color/image_raw",
                  depth_topic=None, width=640, height=480, fps=30,
                  enable_depth=True, depth_max_range=4.0):
    """
    mode: "ros" 或 "realsense"
    """
    if mode == "ros":
        return RosFrameSource(color_topic=color_topic,
                              depth_topic=depth_topic,
                              depth_max_range=depth_max_range)
    if mode == "realsense":
        return RealSenseSource(width=width, height=height, fps=fps,
                               enable_depth=enable_depth,
                               depth_max_range=depth_max_range)
    raise ValueError("未知取流方式: %s" % mode)
