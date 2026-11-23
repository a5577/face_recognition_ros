#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roslaunch 启动管理器 —— 让网页打开时自动拉起 ROS 节点。

用途：Streamlit 页面一被打开（或点按钮），就用 subprocess 在后台执行
`roslaunch ...`，省掉每次手动开终端 source 环境的步骤。

设计要点：

* **进程在 Streamlit 进程之外**：用 `subprocess.Popen` + 新会话(start_new_session)
  启动，停止时对整个进程组发 SIGINT，roslaunch 会按自己的流程关闭所有子节点。
* **幂等**：同一目标只启动一次；若话题上已经有发布者（说明别处已经跑了
  roslaunch），则直接复用，不重复启动，避免相机器材被占两次。
* **持久**：状态保存在模块级变量里。Streamlit 每次交互都会重跑 app.py 脚本，
  但已 import 的模块不会重新加载，所以进程不会被反复拉起。
* **日志**：每个目标的 stdout/stderr 落到日志文件，网页里可以直接看尾部。

预设目标（可用环境变量覆盖，见 config.py）：

* ``camera``    —— realsense2_camera 相机驱动（网页自己跑识别，推荐）
* ``face_node`` —— face_recognition_ros 的完整识别节点 launch
* 自定义        —— 设置 FACE_WEBAPP_LAUNCH_CMD 为任意命令
"""

import os
import signal
import subprocess
import time

import config


# ============================================================
# 预设目标
# ============================================================

PRESETS = {
    "camera": {
        "label": "RealSense 相机驱动（realsense2_camera）",
        "cmd": [
            "roslaunch", "realsense2_camera", "rs_camera.launch",
            "align_depth:=true",
            "color_width:=640", "color_height:=480", "color_fps:=30",
        ],
        "topic": "/camera/color/image_raw",
        "desc": "只启动相机驱动，识别由网页完成（内存占用小，推荐）",
    },
    "face_node": {
        "label": "人脸识别节点（face_recognition.launch）",
        "cmd": [
            "roslaunch", "face_recognition_ros", "face_recognition.launch",
            "vis:=none",
        ],
        "topic": "/camera/color/image_raw",
        "desc": "同时启动相机驱动 + ROS 识别节点（含 YOLO，约 2GB 内存，"
                "不要与网页「🎯 完整模式」同时用）",
    },
}

if config.LAUNCH_CMD:
    PRESETS["custom"] = {
        "label": "自定义命令（FACE_WEBAPP_LAUNCH_CMD）",
        "cmd": config.LAUNCH_CMD,
        "topic": config.LAUNCH_TOPIC,
        "desc": "来自环境变量 FACE_WEBAPP_LAUNCH_CMD",
    }


def available_targets():
    """返回可选目标列表：[{"target": key, "label": ..., "desc": ...}, ...]"""
    return [
        {"target": k, "label": v["label"], "desc": v.get("desc", "")}
        for k, v in PRESETS.items()
    ]


def target_label(target):
    info = PRESETS.get(target)
    return info["label"] if info else str(target)


# ============================================================
# 进程管理
# ============================================================

# target -> {"proc": Popen, "cmd": [...], "log": path, "started_at": ts}
_state = {}


def _log_path(target):
    return os.path.join(config.LAUNCH_LOG_DIR, "%s.log" % target)


def _wrapper_argv(cmd_list):
    """
    包一层 bash：先 source ROS 与工作空间环境，再 exec 目标命令。
    用 "$@" 传参，避免把用户命令拼进字符串里。
    """
    script = 'source "%s" >/dev/null 2>&1 || exit 127;' % config.ROS_SETUP
    if config.WS_SETUP:
        script += ' [ -f "%s" ] && source "%s" >/dev/null 2>&1;' % (
            config.WS_SETUP, config.WS_SETUP)
    script += ' exec "$@"'
    return ["bash", "-c", script, "roslaunch-wrapper"] + list(cmd_list)


def start(target):
    """启动目标对应的 roslaunch（若已在运行则原样返回）。"""
    if target not in PRESETS:
        raise ValueError("未知的启动目标: %s（可选: %s）"
                         % (target, ", ".join(PRESETS)))
    if is_running(target):
        return _state[target]["proc"]

    os.makedirs(config.LAUNCH_LOG_DIR, exist_ok=True)
    log = _log_path(target)
    cmd = PRESETS[target]["cmd"]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    with open(log, "ab", buffering=0) as fh:
        fh.write(("\n===== %s | 启动: %s =====\n"
                  % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
                 .encode("utf-8"))
        proc = subprocess.Popen(
            _wrapper_argv(cmd),
            stdout=fh, stderr=fh, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env, cwd=os.path.expanduser("~"),
        )

    _state[target] = {"proc": proc, "cmd": cmd, "log": log,
                      "started_at": time.time()}
    return proc


def is_running(target):
    entry = _state.get(target)
    if entry is None:
        return False
    if entry["proc"].poll() is None:
        return True
    return False


def stop(target, timeout=8.0):
    """停止目标进程组（SIGINT -> SIGKILL），返回是否执行过停止动作。"""
    entry = _state.get(target)
    if entry is None:
        return False
    proc = entry["proc"]
    if proc.poll() is not None:
        return False

    # 整个会话组一起收：roslaunch 收到 SIGINT 会自己关掉所有子节点
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            try:
                proc.send_signal(sig)
            except Exception:
                pass
        try:
            proc.wait(timeout=wait)
            break
        except Exception:
            continue
    return True


def stop_all():
    for target in list(_state):
        try:
            stop(target)
        except Exception:
            pass


def started_by_app(target):
    """本进程启动且当前仍存活的目标。"""
    return is_running(target)


def uptime(target):
    entry = _state.get(target)
    if entry is None or not is_running(target):
        return 0.0
    return time.time() - entry["started_at"]


def log_tail(target, n_lines=40):
    """返回日志末尾若干行（用于网页里显示启动失败原因）。"""
    path = _log_path(target)
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.read().splitlines()
    except Exception:
        return "（暂无日志）"
    return "\n".join(lines[-int(n_lines):]) or "（暂无日志）"


# ============================================================
# ROS 话题探测
# ============================================================

def ros_core_ready():
    """ROS master 是否可达。"""
    try:
        import rosgraph
        return rosgraph.is_master_online()
    except Exception:
        return False


def topic_publishers(topic):
    """返回话题的发布者节点列表（无 master / 无发布者时为空列表）。"""
    try:
        import rosgraph
        state = rosgraph.Master("/").getSystemState()
        for name, nodes in state[0]:
            if name == topic:
                return list(nodes)
    except Exception:
        pass
    return []


def wait_for_topic(topic, timeout=25.0, poll=0.5):
    """等待话题出现发布者；返回是否等到。"""
    if not topic:
        return True
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if topic_publishers(topic):
            return True
        time.sleep(poll)
    return False


# ============================================================
# 对外主入口
# ============================================================

def ensure_started(target="camera", timeout=25.0, block=True):
    """
    确保目标已启动（幂等）。

    返回 (ok, message)：
        ok=True  已启动 / 已在运行 / 别处已启动（可复用）
        ok=False 启动失败（message 里带日志尾部，便于排查）
    """
    info = PRESETS.get(target)
    if info is None:
        return False, "未知的启动目标: %s" % target

    if is_running(target):
        return True, "已在运行（pid %d）" % _state[target]["proc"].pid

    topic = info.get("topic")

    # 话题上已经有发布者 → 说明外部已经跑过 roslaunch，直接复用
    if topic and topic_publishers(topic):
        return True, "检测到 %s 已有发布者，跳过启动" % topic

    start(target)
    proc = _state[target]["proc"]

    # 启动后短暂观察，命令写错会立刻退出
    time.sleep(1.0)
    if proc.poll() is not None:
        return False, ("启动即退出（exit %s）：\n%s"
                       % (proc.returncode, log_tail(target, 20)))

    if not block:
        return True, "已启动（pid %d）" % proc.pid

    if not wait_for_topic(topic, timeout):
        if proc.poll() is not None:
            return False, ("进程已退出（exit %s）：\n%s"
                           % (proc.returncode, log_tail(target, 20)))
        return False, ("已启动（pid %d），但 %s 超时没有数据，日志：\n%s"
                       % (proc.pid, topic, log_tail(target, 20)))

    return True, "已启动（pid %d），%s 就绪" % (proc.pid, topic)
