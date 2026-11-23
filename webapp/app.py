#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别网页应用（Streamlit）

提供：人脸录入 / 人脸库查看 / 人脸删除 / 识别测试。
复用 InsightFace(buffalo_l) 与 ROS 节点相同的特征数据库，网页端的增删改
会直接写入 face_data/ 与 model/*.pkl，ROS 节点重启后即可生效。

启动：
    cd /home/xiangsi/catkin_ws/src/face_recognition_ros/webapp
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""

import os
import sys
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# 保证可以 import 同目录的 config / face_core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config          # noqa: E402
import face_core as fc  # noqa: E402
import camera          # noqa: E402
import recognizer      # noqa: E402
import live_stream     # noqa: E402
import launch_manager as lm  # noqa: E402


# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="人脸识别管理平台",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 自定义样式（让界面更好看、按钮更大）
# ============================================================

_CSS = """
<style>
:root {
  --brand: #2E7D32;
  --brand-light: #43A047;
  --ink: #1B3A1E;
  --muted: #5B7A5E;
  --radius: 14px;
}

/* 整体背景与内容宽度 */
.stApp {
  background: linear-gradient(180deg, #EFF6F0 0%, #FFFFFF 260px);
}

/* 全局字号：Streamlit 内部基本都用 rem，改根字号即可整体放大。
   想再大/再小就改这一行（16px 是 Streamlit 默认值）。 */
html, body {
  font-size: 18px;
}

.block-container {
  /* 顶部要留够：Streamlit 的工具栏(Deploy 那一行)是悬浮固定的，
     留白不够会把第一屏的标题压在下面。 */
  padding-top: 6rem;
  padding-bottom: 3rem;
  max-width: 1500px;
}
@media (max-width: 640px) {
  .block-container { padding-top: 4.5rem; }
}
/* 顶栏本身透明（默认是白色不透明），避免顶部出现一条白带 */
header[data-testid="stHeader"], [data-testid="stAppHeader"] {
  background: transparent;
}

/* 标题文字 */
h1, h2, h3, h4 { color: var(--ink); letter-spacing: .2px; }

/* ---------- 按钮：加大、圆角、悬停反馈 ---------- */
div.stButton > button, div.stDownloadButton > button {
  border-radius: var(--radius);
  padding: 0.66rem 1.3rem;
  font-size: 1.02rem;
  font-weight: 600;
  min-height: 2.9rem;
  border: 1.5px solid rgba(46, 125, 50, .30);
  background: #FFFFFF;
  color: var(--ink);
  transition: transform .12s ease, box-shadow .12s ease, color .12s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  transform: translateY(-1px);
  border-color: var(--brand);
  color: var(--brand);
  box-shadow: 0 6px 16px rgba(46, 125, 50, .18);
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"],
div.stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
  color: #FFFFFF;
  border: none;
  box-shadow: 0 4px 14px rgba(46, 125, 50, .28);
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
  color: #FFFFFF;
  box-shadow: 0 8px 22px rgba(46, 125, 50, .38);
}
div.stButton > button:disabled {
  opacity: .45;
  box-shadow: none;
  transform: none;
}

/* ---------- 侧边栏 ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #F1F7F2 0%, #FFFFFF 60%);
  border-right: 1px solid rgba(46, 125, 50, .12);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  border-radius: 10px;
  padding: .5rem .7rem;
  font-size: 1.05rem;
  transition: background .12s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(46, 125, 50, .09);
}

/* ---------- 指标卡 ---------- */
div[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid rgba(46, 125, 50, .16);
  border-radius: var(--radius);
  padding: 16px 18px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, .04);
}
div[data-testid="stMetricValue"] { color: var(--brand); font-weight: 700; }

/* ---------- 容器 / 展开面板 ---------- */
div[data-testid="stExpander"] {
  border-radius: var(--radius) !important;
  border: 1px solid rgba(46, 125, 50, .16) !important;
  background: #FFFFFF;
}
details[open] summary { font-weight: 600; }

/* ---------- 图片圆角 ---------- */
div[data-testid="stImage"] img { border-radius: 12px; }

/* ---------- 标签页 ---------- */
button[data-baseweb="tab"] {
  font-size: 1rem;
  font-weight: 600;
  padding: 10px 18px;
}

/* ---------- 输入控件圆角 ---------- */
div[data-baseweb="input"], div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] {
  border-radius: 10px !important;
}

/* ---------- 提示条圆角 ---------- */
div[data-testid="stAlert"] { border-radius: 12px; }

/* ---------- 字号：整体再放大一档 ----------
   根字号已经把 rem 全放大；这里再把原本偏小的文字（说明、标签、
   按钮、标签页、输入框）单独提一点，避免大小反差过大。 */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  font-size: 1.02rem;
}
[data-testid="stCaptionContainer"] small,
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] div {
  font-size: 0.95rem;
}
label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p {
  font-size: 1.03rem;
}
div.stButton > button, div.stDownloadButton > button { font-size: 1.06rem; }
button[data-baseweb="tab"] { font-size: 1.08rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  font-size: 1.12rem;
}
div[data-testid="stMetricValue"] { font-size: 2rem; }
div[data-testid="stMetricLabel"] { font-size: 1.02rem; }
div[data-testid="stAlert"], div[data-testid="stAlert"] p { font-size: 1.02rem; }
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] div { font-size: 1rem; }
code, pre { font-size: 0.95rem; }
</style>
"""


def _inject_css():
    """注入自定义样式（每个页面渲染时调用一次）。"""
    st.markdown(_CSS, unsafe_allow_html=True)


_HEADER_TMPL = """
<div style="display:flex;align-items:center;gap:16px;margin:8px 0 18px 0">
  <div style="font-size:2.5rem;line-height:1">__ICON__</div>
  <div>
    <div style="font-size:1.85rem;font-weight:700;color:#1B3A1E;line-height:1.25">
      __TITLE__</div>
    <div style="color:#5B7A5E;font-size:1rem;margin-top:4px">__DESC__</div>
  </div>
</div>
"""


def _page_header(icon, title, desc=""):
    """页面标题区（比 st.title 更好看）。"""
    st.markdown(
        _HEADER_TMPL.replace("__ICON__", icon)
                    .replace("__TITLE__", title)
                    .replace("__DESC__", desc),
        unsafe_allow_html=True,
    )


# ============================================================
# 模型加载（缓存，只加载一次）
# ============================================================

@st.cache_resource(show_spinner=False)
def _load_model():
    return fc.get_analyzer()


def ensure_model():
    """确保模型已加载；失败时返回 (False, 错误信息)。"""
    if "model_ok" in st.session_state:
        return st.session_state["model_ok"], st.session_state.get("model_err", "")
    try:
        with st.spinner("正在加载人脸识别模型（buffalo_l），首次加载约需 10 秒..."):
            _load_model()
        st.session_state["model_ok"] = True
        st.session_state["model_err"] = ""
    except Exception as exc:  # noqa: BLE001
        st.session_state["model_ok"] = False
        st.session_state["model_err"] = str(exc)
    return st.session_state["model_ok"], st.session_state.get("model_err", "")


# ============================================================
# 小工具
# ============================================================

def load_db():
    return fc.load_db()


def show_image(bgr, caption=None, fit_column=True):
    """
    显示 BGR 图像。

    默认缩放到所在列宽（``fit_column=True``）。传 False 时按原始像素显示，
    640×480 之类的照片会撑出列外，一般只在需要 1:1 查看时才用。
    """
    st.image(bgr, channels="BGR", caption=caption,
             use_column_width=bool(fit_column))


def _thumb(bgr, size=220, pad_color=(32, 32, 32)):
    """
    生成 size×size 的正方形缩略图：等比缩放 + 居中填充黑边。

    网格排版用它的原因：照片尺寸/长宽比各不相同（相机 640×480、手机竖图…），
    直接铺进列里会让每张高度不同、每行对不齐，看起来很乱。统一成正方形后
    每行每列都一样大，而且不裁切、不变形。
    """
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if not h or not w:
        return bgr
    size = int(size)
    scale = size / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=bgr.dtype)
    canvas[:, :] = pad_color
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _grid(items, n_cols=4, gap="small"):
    """
    按行分块产出 ``(column, item)``。

    直接用 ``cols[i % n]`` 的问题是：一行放满后第 n+1 张会往同一列下面堆，
    列与列高度错开，看起来就很乱。按行分块能保证每行都是整齐的一排。
    """
    items = list(items)
    n_cols = max(1, int(n_cols))
    for start in range(0, len(items), n_cols):
        cols = st.columns(n_cols, gap=gap)
        for col, item in zip(cols, items[start:start + n_cols]):
            yield col, item


# 缩略图显示大小 -> 占所在单元格宽度的比例
THUMB_SCALES = {"小": 0.55, "中": 0.75, "大": 1.0}


def _image_slot(scale=1.0, container=st):
    """
    返回一个按比例居中收窄的图片槽位。

    网格里图片都按列宽铺满时，“缩略图大小”就没法生效；
    这里用两侧留白的方式把图片限制在单元格宽度的 scale 倍，控制才真的有效。
    """
    scale = min(1.0, max(0.2, float(scale)))
    if scale >= 0.99:
        return container
    pad = (1.0 - scale) / 2.0
    return container.columns([pad, scale, pad])[1]


def banner_ok(msg):
    st.success(msg)


def banner_warn(msg):
    st.warning(msg)


def banner_err(msg):
    st.error(msg)


# ============================================================
# roslaunch 自动启动（打开网页即运行）
# ============================================================

def launch_status_text(target):
    """当前启动目标的文字状态。"""
    if lm.started_by_app(target):
        return "✅ 运行中（网页启动，已 %.0f 秒）" % lm.uptime(target)
    if lm.ros_core_ready():
        topic = lm.PRESETS[target].get("topic")
        if topic and lm.topic_publishers(topic):
            return "✅ 检测到外部已启动（话题 %s 有数据）" % topic
        return "⚠️ ROS 在线，但目标未由本页启动"
    return "⚪ 未启动（ROS master 未运行）"


def do_launch(target, timeout=25.0, silent=False):
    """启动目标并等待就绪，把结果写进 session_state。"""
    try:
        with st.spinner("正在启动 %s ..." % lm.target_label(target)):
            ok, msg = lm.ensure_started(target, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        ok, msg = False, "启动失败：%s" % exc
    st.session_state["launch_msg"] = ("ok:" if ok else "err:") + msg
    if ok:
        if not silent:
            banner_ok(msg)
    elif not silent:
        banner_err(msg)
    return ok


def maybe_autolaunch():
    """打开网页时自动启动 roslaunch（只在本进程内启动一次）。"""
    if not st.session_state.get("launch_auto"):
        return
    target = st.session_state.get("launch_target", "camera")
    if target not in lm.PRESETS:
        return
    if lm.started_by_app(target):
        return
    # 外部已经跑过 roslaunch（话题有发布者）时直接复用，不再重复启动
    topic = lm.PRESETS[target].get("topic")
    if lm.ros_core_ready() and topic and lm.topic_publishers(topic):
        return
    # 首次打开时最多等 15 秒：起来了就直接可用，起不来也不会让页面卡太久
    do_launch(target, timeout=15.0, silent=True)


def render_launch_panel(container, key_prefix="sb"):
    """roslaunch 控制面板（侧边栏 / 实时识别页共用）。"""
    targets = list(lm.PRESETS.keys())
    target = container.selectbox(
        "启动目标", targets, key=key_prefix + "_launch_target",
        index=targets.index(st.session_state.get("launch_target", targets[0]))
        if st.session_state.get("launch_target") in targets else 0,
        format_func=lambda k: lm.PRESETS[k]["label"],
    )
    st.session_state["launch_target"] = target
    container.caption(lm.PRESETS[target].get("desc", ""))
    container.caption("状态：" + launch_status_text(target))

    c1, c2 = container.columns(2)
    running = lm.started_by_app(target)
    if c1.button("▶️ 启动", key=key_prefix + "_launch_start",
                 disabled=running, use_container_width=True):
        do_launch(target)
    if c2.button("⏹ 停止", key=key_prefix + "_launch_stop",
                 disabled=not running, use_container_width=True):
        stopped = lm.stop(target)
        st.session_state["launch_msg"] = (
            "ok:已停止" if stopped else "err:没有由本页启动的进程")
        st.rerun()

    # 注意：外层可能已经是 expander，Streamlit 不允许 expander 嵌套，
    # 所以这里用复选框展开日志。
    if container.checkbox("📜 显示启动日志", key=key_prefix + "_launch_log"):
        container.code(lm.log_tail(target, 30), language=None)
        container.caption("日志文件：%s" % os.path.join(
            config.LAUNCH_LOG_DIR, "%s.log" % target))
    return target


# ============================================================
# 侧边栏
# ============================================================

def render_sidebar():
    st.sidebar.markdown(
        '<div style="font-size:1.45rem;font-weight:700;color:#1B3A1E;'
        'padding:8px 2px 14px 2px">🙂 人脸识别平台</div>',
        unsafe_allow_html=True)
    page = st.sidebar.radio(
        "功能导航",
        ["📊 概览", "➕ 人脸录入", "🗂️ 人脸库管理",
         "📹 实时识别", "🖼️ 本次任务结果", "🔍 识别测试"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    with st.sidebar.expander("⚙️ 参数与路径", expanded=False):
        st.caption("照片目录")
        st.code(config.FACE_DATA_DIR, language=None)
        st.caption("陌生人截图目录")
        st.code(config.STRANGER_DIR, language=None)
        st.caption("识别画面存档目录")
        st.code(config.RECOGNITION_SNAPSHOT_DIR, language=None)
        st.caption("特征数据库")
        st.code(config.DB_FILE, language=None)
        st.caption("模型 / 后端")
        st.code("%s  [%s]" % (config.MODEL_NAME, ", ".join(config.PROVIDERS)),
                language=None)
        st.caption("默认阈值 %.2f（与 ROS 节点 params.yaml 一致）" % config.THRESHOLD)

    st.sidebar.divider()
    with st.sidebar.expander("🚀 ROS 启动（roslaunch）", expanded=True):
        st.checkbox("打开网页时自动启动", key="launch_auto",
                    help="勾选后，每次打开/刷新页面都会确保 roslaunch 已运行"
                         "（已在运行时不会重复启动）。")
        render_launch_panel(st, key_prefix="sb")

    db = load_db()
    n_people, n_emb = fc.db_stats(db)
    st.sidebar.divider()
    st.sidebar.metric("已录入人数", n_people)
    st.sidebar.metric("特征总数", n_emb)
    return page


# ============================================================
# 页面 1：概览
# ============================================================

def page_overview():
    _page_header("📊", "概览", "人脸库总览 · 照片目录 %s" % config.FACE_DATA_DIR)

    db = load_db()
    n_people, n_emb = fc.db_stats(db)

    c1, c2, c3 = st.columns(3, gap="large")
    c1.metric("已录入人数", n_people)
    c2.metric("照片数", len(sum([fc.list_person_images(n) for n in fc.list_people()],
                                [])) if n_people else 0)
    c3.metric("特征总数", n_emb)

    st.divider()
    if n_people == 0:
        st.info("人脸库还是空的。请到「➕ 人脸录入」页添加第一张人脸。")
        return

    st.subheader("👥 成员一览")
    for col, (name, entry) in _grid(sorted(db.items()), 4, gap="large"):
        files = fc.list_person_images(name)
        with col:
            with st.container(border=True):
                if files:
                    img = fc.imread(os.path.join(fc.person_dir(name), files[0]))
                    if img is not None:
                        st.image(_thumb(img, 260), channels="BGR",
                                 use_column_width=True)
                st.markdown("##### %s" % name)
                st.caption("照片 %d 张 · 特征 %d 条"
                           % (len(files), len(entry.get("embeddings", []))))


# ============================================================
# 页面 2：人脸录入
# ============================================================

_state_defaults = {
    "last_enroll_results": None,
    "last_enroll_name": None,
    "live_src": None,
    "live_src_key": None,
    "live_rec": None,
    "live_rec_key": None,
    "current_snapshot_task": None,
    "current_snapshot_task_label": "",
    "current_stranger_task": None,
    "enroll_src": None,
    "enroll_src_key": None,
    "enroll_caps": [],
    "launch_auto": bool(config.AUTOLAUNCH),
    "launch_target": (config.LAUNCH_TARGET
                      if config.LAUNCH_TARGET in lm.PRESETS else "camera"),
    "launch_msg": "",
}


def _init_state():
    for k, v in _state_defaults.items():
        st.session_state.setdefault(k, v)


def _close_enroll_source():
    src = st.session_state.get("enroll_src")
    if src is not None:
        try:
            src.close()
        except Exception:
            pass
    st.session_state["enroll_src"] = None
    st.session_state["enroll_src_key"] = None


def _get_enroll_source(mode, kwargs):
    """录入页专用的相机取流（按配置复用，配置变化时重建）。"""
    key = repr((mode, sorted(kwargs.items())))
    if (st.session_state.get("enroll_src") is not None
            and st.session_state.get("enroll_src_key") == key):
        return st.session_state["enroll_src"]
    _close_enroll_source()
    src = camera.create_source(mode, **kwargs)
    st.session_state["enroll_src"] = src
    st.session_state["enroll_src_key"] = key
    return src


def _render_enroll_results(name, results):
    st.divider()
    ok = [r for r in results if r["status"] == "ok"]
    skip = [r for r in results if r["status"] != "ok"]
    st.subheader("录入结果")
    st.write("✅ 成功 **%d** 张，⚠️ 跳过 **%d** 张" % (len(ok), len(skip)))

    if skip:
        for r in skip:
            banner_warn("%s：%s" % (r["file"], r["reason"]))

    if ok:
        st.caption("以下人脸已写入 %s" % fc.person_dir(name))
        for col, r in _grid(ok, 4, gap="small"):
            path = os.path.join(fc.person_dir(name), r["saved_as"])
            img = fc.imread(path)
            with col:
                with st.container(border=True):
                    if img is not None:
                        if r.get("bbox"):
                            img = fc.draw_faces(
                                img,
                                [{"bbox": r["bbox"], "det_score": r["det_score"]}],
                            )
                        st.image(_thumb(img, 240), channels="BGR",
                                 use_column_width=True)
                    st.caption("检测分 %.2f · 人脸 %d"
                               % (r["det_score"], r["n_faces"]))


def page_enroll():
    _page_header("➕", "人脸录入 / 添加",
                 "上传照片、D435i 抓拍或浏览器拍照，自动检测人脸并提取特征")

    db = load_db()
    people = sorted(db.keys())

    # ---------- 选择身份 ----------
    mode = st.radio(
        "录入方式",
        ["新建身份", "追加到已有身份"] if people else ["新建身份"],
        horizontal=True,
    )

    name = ""
    if mode == "追加到已有身份":
        name = st.selectbox("选择已有身份", people)
    else:
        name = st.text_input("身份名称（姓名 / ID）", "",
                             placeholder="例如：chenzhen").strip()

    # ---------- 选择图片来源 ----------
    tab_upload, tab_d435i, tab_camera = st.tabs(
        ["📁 上传图片", "📹 D435i 抓拍", "📷 浏览器拍照"])
    image_items = []

    with tab_upload:
        files = st.file_uploader(
            "选择一张或多张图片（jpg / png / bmp / webp）",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        for f in files or []:
            image_items.append((f.name, fc.decode_image(f.getvalue())))

    # ---- 直接从 D435i 抓拍：不依赖浏览器摄像头权限 ----
    with tab_d435i:
        st.caption("直接从深度相机抓拍录入，不需要浏览器摄像头权限（推荐）。"
                   "可先开预览看画面，再抓拍。")
        c1, c2 = st.columns([1, 2])
        src_label = c1.radio("取流方式", ["ROS 话题", "RealSense 直连"],
                             horizontal=True, key="enroll_src_mode")
        src_mode = "ros" if src_label.startswith("ROS") else "realsense"
        src_kwargs = {}
        if src_mode == "ros":
            src_kwargs["color_topic"] = c2.text_input(
                "彩色图像话题", "/camera/color/image_raw", key="enroll_topic")
        else:
            src_kwargs.update(width=640, height=480, fps=30, enable_depth=False)
            c2.caption("直连模式需先安装 pyrealsense2，且不能与 ROS 驱动同时占用相机")

        previewing = live_stream.is_running()
        psize = _preview_size_selector(st, key="enroll_preview_size",
                                       label="预览尺寸")
        b1, b2, b3 = st.columns(3, gap="medium")
        start_prev = b1.button("🔴 开启预览", key="enroll_preview_on",
                               disabled=previewing, use_container_width=True)
        snap_btn = b2.button("📸 抓拍一张", key="enroll_snap",
                             type="primary", use_container_width=True)
        stop_prev = b3.button("⏹ 停止预览", key="enroll_preview_off",
                              disabled=not previewing, use_container_width=True)

        if start_prev:
            try:
                prev_src = camera.create_source(src_mode, **src_kwargs)
                # analyze=None -> 纯预览，不跑识别，省 CPU
                live_stream.start_pipeline(prev_src, None, None,
                                           port=8502, jpeg_quality=75,
                                           max_fps=15)
                previewing = True
                banner_ok("相机预览已开启。")
            except Exception as exc:  # noqa: BLE001
                banner_err("连接相机失败：%s" % exc)
                if src_mode == "ros" and not camera.ros_core_ready():
                    st.info("ROS master 未运行，请先启动相机驱动：")
                    st.code("source /opt/ros/noetic/setup.bash\n"
                            "roslaunch realsense2_camera rs_camera.launch "
                            "align_depth:=true", language="bash")

        if stop_prev:
            live_stream.stop_pipeline()
            _close_enroll_source()
            previewing = False
            banner_ok("已停止预览。")

        if previewing:
            _render_mjpeg(8502, size=psize)
            st.caption("画面为相机实时预览；点「📸 抓拍一张」会把当前画面加入待录入。")

        if snap_btn:
            color = None
            if live_stream.is_running():
                color = live_stream.grab_frame()          # 从预览画面取当前帧
            if color is None:
                try:                                      # 没开预览就单独抓一帧
                    src = _get_enroll_source(src_mode, src_kwargs)
                    color, _, _ = src.read(timeout=5.0)
                except Exception as exc:  # noqa: BLE001
                    banner_err("连接相机失败：%s" % exc)
                    color = None
            if color is None:
                banner_warn("未收到图像数据，请确认相机已启动、话题名正确。")
            else:
                ts = time.strftime("d435i_%Y%m%d_%H%M%S.jpg")
                st.session_state["enroll_caps"].append((ts, color))
                banner_ok("已抓拍一张，待录入 %d 张。"
                          % len(st.session_state["enroll_caps"]))

    with tab_camera:
        shot = st.camera_input("用浏览器摄像头拍一张照片")
        st.caption("⚠️ 浏览器只在 **HTTPS 或 localhost** 下允许调用摄像头。"
                   "若你在别的电脑上用 `http://<IP>:8501` 打开，浏览器会拦截。"
                   "解决办法见页面下方「打不开摄像头？」说明，"
                   "或改用上面的「📹 D435i 抓拍」。")
        if shot is not None:
            ts = time.strftime("camera_%Y%m%d_%H%M%S.jpg")
            image_items.append((ts, fc.decode_image(shot.getvalue())))

    # ---- D435i 已抓拍的照片 ----
    caps = st.session_state.get("enroll_caps") or []
    if caps:
        h1, h2 = st.columns([3, 1], vertical_alignment="bottom")
        h1.markdown("**D435i 已抓拍 %d 张**" % len(caps))
        if h2.button("🗑️ 清空抓拍", key="clear_caps", use_container_width=True):
            st.session_state["enroll_caps"] = []
            st.rerun()
        for col, (idx, (cname, cimg)) in _grid(list(enumerate(caps)), 4):
            with col:
                with st.container(border=True):
                    st.image(_thumb(cimg, 200), channels="BGR",
                             use_column_width=True)
                    st.caption(cname)
                    if st.button("🗑️ 删除", key="del_cap_%d" % idx,
                                 use_container_width=True):
                        st.session_state["enroll_caps"].pop(idx)
                        st.rerun()
        image_items += list(caps)

    # ---------- 录入选项 ----------
    with st.expander("高级选项", expanded=False):
        require_single = st.checkbox(
            "只接受每张图片中检测到的最大人脸（多脸时跳过）", value=False)
        min_det_score = st.slider("最小人脸检测分", 0.0, 1.0, 0.0, 0.05,
                                  help="低于该分数的照片会被跳过，用于过滤模糊/侧脸")

    st.divider()
    disable = not name or not image_items
    if st.button("✅ 开始录入", type="primary", disabled=disable,
                 use_container_width=True):
        if not name:
            banner_err("请先填写身份名称。")
        elif not image_items:
            banner_err("请先上传或拍摄至少一张图片。")
        else:
            with st.spinner("正在提取人脸特征..."):
                db, results = fc.enroll_images(
                    name, image_items,
                    require_single=require_single,
                    min_det_score=min_det_score,
                )
            st.session_state["last_enroll_results"] = results
            st.session_state["last_enroll_name"] = name
            st.session_state["enroll_caps"] = []      # 已录入，清空待录入抓拍

    if st.session_state.get("last_enroll_results"):
        _render_enroll_results(
            st.session_state["last_enroll_name"],
            st.session_state["last_enroll_results"],
        )
    elif disable:
        if not name:
            st.info("请填写身份名称。")
        elif not image_items:
            st.info("请至少上传或拍摄一张图片。")

    # ---------- 浏览器摄像头打不开的解决办法 ----------
    with st.expander("❓ 打不开摄像头？", expanded=False):
        st.markdown(
            "浏览器出于安全考虑，**只允许 HTTPS 或 localhost 页面调用摄像头**。\n\n"
            "**方法 1（最简单）**：在你正在用的那台电脑上，把地址换成 "
            "`http://localhost:8501`。\n\n"
            "**方法 2（浏览器在另一台电脑上）**：做一次 SSH 端口转发，"
            "然后浏览器访问 `http://localhost:8501`：\n\n"
            "```bash\n"
            "ssh -L 8501:localhost:8501 -L 8502:localhost:8502 用户名@<服务器IP>\n"
            "```\n\n"
            "**方法 3**：给 Streamlit 配 HTTPS 自签名证书：\n\n"
            "```bash\n"
            "mkdir -p ~/.streamlit/certs && cd ~/.streamlit/certs\n"
            "openssl req -x509 -newkey rsa:2048 -nodes -days 365 \\\n"
            "  -keyout key.pem -out cert.pem -subj \"/CN=localhost\"\n"
            "streamlit run app.py \\\n"
            "  --server.sslCertFile=$HOME/.streamlit/certs/cert.pem \\\n"
            "  --server.sslKeyFile=$HOME/.streamlit/certs/key.pem\n"
            "```\n"
            "然后用 `https://<IP>:8501` 访问，并在浏览器里选择“继续前往”。\n\n"
            "**方法 4（Chrome）**：地址栏打开 `chrome://flags/#unsafely-treat-insecure-origin-as-secure`，"
            "把 `http://<你的IP>:8501` 加进白名单并重启浏览器。\n\n"
            "**方法 5（推荐，绕开问题）**：改用上面的「📹 D435i 抓拍」，"
            "直接从深度相机抓拍录入，完全不依赖浏览器摄像头权限。\n\n"
            "另外请检查：地址栏左侧的权限图标是否把摄像头设成了“阻止”，"
            "以及系统隐私设置里是否允许浏览器使用摄像头。"
        )


# ============================================================
# 页面 3：人脸库管理（查看 / 删除 / 重命名 / 重建）
# ============================================================

def page_manage():
    _page_header("🗂️", "人脸库管理",
                 "查看、删除、重命名人脸，以及按照片重建特征")

    db = load_db()
    if not db:
        st.info("人脸库为空，暂无可管理的内容。")
        return

    # ---------- 全局操作 ----------
    with st.expander("🔧 全局操作", expanded=False):
        st.caption("按 face_data 目录下的照片重新计算所有人的特征，"
                   "可用于修复数据或导入外部照片后同步。")
        if st.button("按照片重建整个数据库", use_container_width=True):
            with st.spinner("正在重建全部特征..."):
                _, stats = fc.rebuild_all()
            banner_ok("重建完成：%d 个身份。" % len(stats))
            st.rerun()

    st.divider()

    # ---------- 逐人管理 ----------
    for name in sorted(db.keys()):
        entry = db[name]
        files = fc.list_person_images(name)
        n_emb = len(entry.get("embeddings", []))

        with st.expander("**%s**  ·  照片 %d 张 · 特征 %d 条"
                         % (name, len(files), n_emb), expanded=False):

            # ---- 该身份的操作按钮 ----
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1],
                                        vertical_alignment="bottom")
            new_name = c1.text_input("重命名", value=name,
                                     key="rename_%s" % name)
            rename_click = c2.button("重命名", key="rename_btn_%s" % name,
                                     use_container_width=True)
            rebuild_click = c3.button("🔄 重建特征", key="rebuild_%s" % name,
                                      use_container_width=True)
            del_click = c4.button("🗑️ 删除该人", key="del_person_%s" % name,
                                  use_container_width=True)

            if rename_click:
                new_name = new_name.strip()
                if not new_name:
                    banner_err("新名称不能为空。")
                elif new_name != name:
                    try:
                        fc.rename_person(name, new_name)
                        banner_ok("已重命名为 %s" % new_name)
                        st.rerun()
                    except FileExistsError as exc:
                        banner_err(str(exc))

            if rebuild_click:
                with st.spinner("正在重建 %s 的特征..." % name):
                    fc.rebuild_person(name)
                banner_ok("已重建 %s" % name)
                st.rerun()

            if del_click:
                fc.delete_person(name)
                banner_ok("已删除身份 %s" % name)
                st.rerun()

            st.caption("照片目录：`%s`" % fc.person_dir(name))
            st.divider()

            # ---- 照片网格 ----
            if not files:
                st.caption("该身份没有照片文件（仅有特征）。")
                continue

            g1, g2, g3 = st.columns([1, 1, 2], vertical_alignment="bottom")
            n_cols = int(g1.selectbox("每行张数", [2, 3, 4, 5, 6], index=2,
                                      key="gal_cols_%s" % name))
            size_label = g2.selectbox("缩略图大小", list(THUMB_SCALES),
                                      index=1, key="gal_size_%s" % name)
            g3.caption("共 %d 张 · 缩略图统一为正方形（等比缩放 + 留边，不裁切），"
                       "右键「另存为」可看原图" % len(files))

            for col, fname in _grid(files, n_cols, gap="small"):
                path = os.path.join(fc.person_dir(name), fname)
                img = fc.imread(path)
                with col:
                    with st.container(border=True):
                        if img is not None:
                            _image_slot(THUMB_SCALES[size_label]).image(
                                _thumb(img, 320), channels="BGR",
                                use_column_width=True)
                            h, w = img.shape[:2]
                            st.caption("%s · %d×%d" % (fname, w, h))
                        else:
                            st.warning("无法读取")
                            st.caption(fname)
                        if st.button("🗑️ 删除",
                                     key="del_img_%s_%s" % (name, fname),
                                     use_container_width=True):
                            fc.delete_image(name, fname)
                            banner_ok("已删除 %s/%s" % (name, fname))
                            st.rerun()


# ============================================================
# 页面 4：识别测试
# ============================================================

def page_test():
    _page_header("🔍", "识别测试",
                 "上传一张图片，检测人脸并与已录入的人脸库比对（不影响数据库）")

    db = load_db()
    if not db:
        st.info("人脸库为空，请先录入人脸。")
        return

    threshold = st.slider("识别阈值", 0.0, 1.0, float(config.THRESHOLD), 0.01,
                          help="相似度大于等于该值才判为已知身份")

    test_file = st.file_uploader("上传待识别图片",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 key="test_upload")
    if test_file is None:
        return

    bgr = fc.decode_image(test_file.getvalue())
    if bgr is None:
        banner_err("图片读取失败。")
        return

    with st.spinner("正在检测与识别..."):
        faces = fc.detect_faces(bgr)

    if not faces:
        banner_warn("未检测到人脸。")
        show_image(bgr)
        return

    annotated = bgr.copy()
    results = []
    snapshot_infos = []
    stranger_saver = st.session_state.get("test_stranger_saver")
    if stranger_saver is None:
        stranger_saver = fc.StrangerFaceSaver()
        st.session_state["test_stranger_saver"] = stranger_saver
    frame_saver = st.session_state.get("test_frame_saver")
    if frame_saver is None:
        frame_saver = fc.RecognitionFrameSaver()
        st.session_state["test_frame_saver"] = frame_saver
    for f in faces:
        name, score, confirmed = fc.recognize_face(f["embedding"], db,
                                                   threshold=threshold)
        saved_path = None
        if not confirmed:
            saved_path = stranger_saver.save(bgr, f["bbox"], f["embedding"])
        results.append((name, score, confirmed, f, saved_path))
        snapshot_infos.append({
            "bbox": f["bbox"], "name": name, "quality": "ok",
            "confirmed": bool(confirmed),
            "is_stranger": bool(not confirmed),
            "result_ready": True,
            "score": float(score),
            "status": "%s %.2f" % (name, score),
        })
        x1, y1, x2, y2 = [int(round(v)) for v in f["bbox"]]
        color = (0, 200, 0) if confirmed else (0, 0, 255)
        label = ("%s %.2f" % (name, score) if confirmed
                 else "Unknown %.2f" % score)
        import cv2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    snapshot_path, snapshot_category = frame_saver.save(
        annotated, snapshot_infos
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("识别结果")
        show_image(annotated)
    with c2:
        st.subheader("详情")
        for i, (name, score, confirmed, f, saved_path) in enumerate(results, 1):
            tag = "✅" if confirmed else "❓"
            st.write("%s 人脸 %d：**%s**  相似度 %.4f  检测分 %.2f"
                     % (tag, i, name, score, f["det_score"]))
            if saved_path:
                st.caption("陌生人脸已保存：%s" % saved_path)
        if any(r[4] for r in results):
            st.caption("陌生人截图目录：%s" % config.STRANGER_DIR)
        if snapshot_path:
            st.caption("识别画面已保存（%s）：%s"
                       % (fc.RECOGNITION_SNAPSHOT_LABELS[snapshot_category],
                          snapshot_path))


# ============================================================
# 页面：本次任务识别结果
# ============================================================

def page_task_results():
    _page_header(
        "🖼️", "本次任务结果",
        "展示当前识别任务保存的图片，以及图片中每张人脸是数据库人员还是陌生人",
    )

    task = st.session_state.get("current_snapshot_task")
    if task is None:
        st.info("当前还没有识别任务。请先到「📹 实时识别」开始视频或抓拍一帧。")
        return

    pipe = live_stream.get_pipeline()
    if (live_stream.is_running() and pipe is not None
            and getattr(pipe, "snapshot_saver", None) is task):
        st.caption("实时识别运行中：新保存的图片和识别结果会自动出现在下方。")
        _render_task_results_live(pipe.server.port)
        return

    events = task.get_events()
    label = st.session_state.get("current_snapshot_task_label") or "识别任务"
    c1, c2, c3 = st.columns(3)
    c1.metric("已保存结果", len(events))
    c2.metric("任务类型", label)
    c3.metric("任务编号", task.task_id)
    st.caption("任务开始：%s · 同一识别结果在一次任务中只保存一次。"
               % task.created_at)

    c1, c2 = st.columns([1, 4])
    if c1.button("🔄 刷新结果", use_container_width=True):
        st.rerun()
    c2.caption("实时识别未运行，当前展示该任务已保存的全部结果。")

    if not events:
        st.info("当前任务还没有已确认的结果。完整模式需要先完成多帧投票；"
                "快速模式得到识别结果后会立即保存。")
        return

    st.divider()
    for col, event in _grid(events, 2, gap="large"):
        with col:
            with st.container(border=True):
                img = fc.imread(event["path"])
                if img is not None:
                    st.image(img, channels="BGR", use_column_width=True)
                st.markdown("##### %s"
                            % fc.RECOGNITION_SNAPSHOT_LABELS[
                                event["category"]
                            ])
                st.caption("%s · %s"
                           % (event["created_at"],
                              os.path.basename(event["path"])))
                for i, result in enumerate(event["results"], 1):
                    if result["is_stranger"]:
                        st.write("🚫 人脸 %d：**陌生人** · 相似度 %.3f"
                                 % (i, result["score"]))
                    else:
                        st.write("✅ 人脸 %d：数据库人员 **%s** · 相似度 %.3f"
                                 % (i, result["name"], result["score"]))
                st.caption(event["path"])


# ============================================================
# 页面 5：实时识别（Intel RealSense D435i）
# ============================================================

def _close_live_source():
    src = st.session_state.get("live_src")
    if src is not None:
        try:
            src.close()
        except Exception:
            pass
    st.session_state["live_src"] = None
    st.session_state["live_src_key"] = None


def _get_live_source(mode, kwargs):
    """按配置获取/复用相机取流对象（配置变化时自动重建）。"""
    key = repr((mode, sorted(kwargs.items())))
    if (st.session_state.get("live_src") is not None
            and st.session_state.get("live_src_key") == key):
        return st.session_state["live_src"]
    _close_live_source()
    src = camera.create_source(mode, **kwargs)
    st.session_state["live_src"] = src
    st.session_state["live_src_key"] = key
    return src


def _close_recognizer():
    st.session_state["live_rec"] = None
    st.session_state["live_rec_key"] = None


def _new_snapshot_task(label):
    """每次开始视频或抓拍时创建一个新的结果保存任务。"""
    task = fc.RecognitionFrameSaver()
    st.session_state["current_snapshot_task"] = task
    st.session_state["current_snapshot_task_label"] = label
    return task


def _new_stranger_task():
    """每次开始识别时创建一个新的陌生人截图任务。"""
    task = fc.StrangerFaceSaver()
    st.session_state["current_stranger_task"] = task
    return task


def _get_recognizer(key, params, stranger_saver=None):
    """按参数获取/复用完整识别器（与 ROS 节点同流程）。"""
    rec = st.session_state.get("live_rec")
    if rec is not None and st.session_state.get("live_rec_key") == key:
        rec.set_params(**params)
        rec.update_db(load_db())
        if stranger_saver is not None:
            rec.set_stranger_saver(stranger_saver)
        return rec
    rec = recognizer.LiveRecognizer(
        params=recognizer.load_params(), db=load_db(),
        stranger_saver=stranger_saver,
    )
    rec.set_params(**params)
    st.session_state["live_rec"] = rec
    st.session_state["live_rec_key"] = key
    return rec


def _fast_analyze(color, db, threshold, min_det, stranger_saver=None):
    """快速模式：整帧直接检测 + 识别，只返回结果（不画图）。"""
    infos = []
    for f in fc.detect_faces(color):
        if f["det_score"] < min_det:
            continue
        name, score, confirmed = fc.recognize_face(f["embedding"], db, threshold)
        saved_path = None
        if not confirmed and stranger_saver is not None:
            saved_path = stranger_saver.save(
                color, f["bbox"], f["embedding"]
            )
        status = ("%s %.2f" % (name, score) if confirmed
                  else "陌生人 %.2f" % score)
        if saved_path:
            status += "（已保存）"
        infos.append({
            "bbox": [int(round(v)) for v in f["bbox"]],
            "name": name, "score": float(score),
            "confirmed": bool(confirmed),
            "is_stranger": bool(not confirmed),
            "stranger_confirmed": bool(not confirmed),
            "result_ready": True,
            "stranger_saved": saved_path,
            "det_score": float(f["det_score"]),
            "quality": "ok", "vote_count": 0,
            "status": status,
            "color": (0, 255, 0) if confirmed else (0, 0, 255),
        })
    return infos


def _fast_process(color, db, threshold, min_det, font=None,
                  stranger_saver=None):
    """快速模式：识别 + 画图，返回 (display, infos)。"""
    infos = _fast_analyze(color, db, threshold, min_det, stranger_saver)
    return recognizer.draw_infos(color.copy(), infos, font), infos


def _render_live_results(infos):
    if not infos:
        return
    st.subheader("本帧识别结果")
    for i, r in enumerate(infos, 1):
        tag = "✅" if r["confirmed"] else (
            "🚫" if r.get("is_stranger") else "❓"
        )
        extra = ""
        if "det_score" in r:
            extra += "  检测分 %.2f" % r["det_score"]
        if r.get("vote_count"):
            extra += "  票数 %d" % r["vote_count"]
        if r.get("quality") not in (None, "ok"):
            extra += "  质量: %s" % r["quality"]
        if r.get("stranger_saved"):
            extra += "  已保存: %s" % r["stranger_saved"]
        st.write("%s 人脸 %d：**%s**  相似度 %.3f%s"
                 % (tag, i, r["name"], r["score"], extra))


def _render_stranger_gallery():
    """只显示当前任务保存的陌生人截图。"""
    task = st.session_state.get("current_stranger_task")
    if task is None:
        st.caption("当前还没有识别任务。请先开始实时视频或抓拍一帧。")
        return

    pipe = live_stream.get_pipeline()
    if (live_stream.is_running() and pipe is not None
            and getattr(pipe, "stranger_saver", None) is task):
        st.caption("仅显示本次任务；实时识别中，新保存的陌生人截图会自动出现。")
        _render_stranger_gallery_live(pipe.server.port)
        return

    events = task.get_events()
    limit_text = ("不限" if task.max_images <= 0
                  else "%d 张" % task.max_images)
    st.caption("仅显示当前任务（%s 开始）保存的陌生人截图；每个不同陌生人"
               "各保存一次，本任务上限：%s；不包含之前的任务。"
               % (task.created_at, limit_text))
    if not events:
        st.caption("本次任务暂时没有陌生人截图。")
        return
    for col, event in _grid(events, 4, gap="small"):
        img = fc.imread(event["path"])
        with col:
            with st.container(border=True):
                if img is not None:
                    st.image(_thumb(img, 220), channels="BGR",
                             use_column_width=True)
                st.caption("%s · %s"
                           % (event["created_at"],
                              os.path.basename(event["path"])))


def _render_recognition_gallery():
    """显示当前任务保存的识别画面，不读取历史任务。"""
    task = st.session_state.get("current_snapshot_task")
    if task is None:
        st.caption("当前还没有识别任务。请先开始实时视频或抓拍一帧。")
        return

    pipe = live_stream.get_pipeline()
    if (live_stream.is_running() and pipe is not None
            and getattr(pipe, "snapshot_saver", None) is task):
        st.caption("仅显示本次任务；识别运行中，新保存的图片会自动出现在下方。")
        _render_task_results_live(pipe.server.port)
        return

    events = task.get_events()
    st.caption("仅显示当前任务（%s 开始）保存的图片和结果，不包含之前的任务。"
               % task.created_at)
    if not events:
        st.caption("本次任务暂时没有已确认的识别画面。")
        return
    for col, event in _grid(events, 2, gap="small"):
        img = fc.imread(event["path"])
        with col:
            with st.container(border=True):
                if img is not None:
                    st.image(img, channels="BGR", use_column_width=True)
                st.caption("%s · %s"
                           % (fc.RECOGNITION_SNAPSHOT_LABELS[
                                  event["category"]
                              ],
                              event["created_at"]))
                for result in event["results"]:
                    if result["is_stranger"]:
                        st.write("🚫 陌生人 · 相似度 %.3f"
                                 % result["score"])
                    else:
                        st.write("✅ 数据库人员 **%s** · 相似度 %.3f"
                                 % (result["name"], result["score"]))


def _render_frame(color, depth, caption="彩色 / 识别"):
    if depth is not None:
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), channels="RGB",
                 use_column_width=True, caption=caption)
        c2.image(depth, channels="BGR", use_column_width=True,
                 caption="深度（伪彩色）")
    else:
        st.image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), channels="RGB",
                 use_column_width=True, caption=caption)


# 预览尺寸预设: 标签 -> (显示方式, 高度px)
#   fit    : 按容器宽度放大，但整帧完整可见（长宽比不变，多余处留黑边）
#   native : 原始像素 1:1 显示（640x480），不放大
#   full   : 铺满容器宽度（可能很高，超出部分需要滚动）
_PREVIEW_SIZES = {
    "自适应宽度（完整画面，推荐）": ("fit", 420),
    "大（高 720px）": ("fit", 720),
    "原始尺寸 1:1（640×480）": ("native", 480),
    "全宽（很高，可能需滚动）": ("full", 900),
}

_PREVIEW_SIZE_HELP = (
    "「自适应宽度」会把整帧缩放到容器宽度内、不裁切；"
    "「原始尺寸」按相机像素 1:1 显示；「全宽」铺满页面宽度。"
)


def _preview_size_selector(container=st, key="preview_size", label="预览尺寸"):
    """预览显示尺寸选择框（录入页 / 实时识别页共用）。"""
    return container.selectbox(
        label, list(_PREVIEW_SIZES.keys()), key=key,
        help=_PREVIEW_SIZE_HELP,
    )


def _render_mjpeg(port, height=540, size=None):
    """
    在页面里嵌入 MJPEG 实时视频流。

    用浏览器当前的 hostname 拼接地址，所以本机 / 局域网访问都能直接播；
    右下状态通过 /status 每秒轮询更新。

    size: _PREVIEW_SIZES 里的标签（决定画面显示尺寸）。
          轮廓之外的控制方式：height 作为 fit 模式的默认高度。
    """
    mode, img_h = _PREVIEW_SIZES.get(size, ("fit", int(height)))

    if mode == "native":
        # 原始像素显示：不放大，容器太窄时才缩小
        img_style = ("width:auto;max-width:100%;height:auto;"
                     "border-radius:10px;background:#111;display:block")
        frame_h = img_h + 46
    elif mode == "full":
        img_style = ("width:100%;height:auto;"
                     "border-radius:10px;background:#111;display:block")
        frame_h = img_h
    else:  # fit：整帧完整可见，不裁切
        # 注意：这里不能做 % 格式化，字符串里的 width:100% 会和占位符冲突
        img_style = ("width:100%;max-height:" + str(int(img_h)) +
                     "px;object-fit:contain;"
                     "border-radius:10px;background:#111;display:block")
        frame_h = img_h + 46

    # 注意：HTML/CSS 里有 % 和 {} 字符，不能用 % 或 format 格式化，用占位符替换
    html = """
    <div style="font-family:sans-serif">
      <img id="livevid" style="__IMGSTYLE__" />
      <div id="livestat" style="font-size:13px;color:#666;margin-top:8px">
        连接中…</div>
    </div>
    <script>
      // components.html 使用 srcdoc iframe；它的 location 可能是 about:srcdoc，
      // 直接读取 window.location.hostname 在远程访问时可能得到空值或 localhost。
      // 优先从外层 Streamlit 页面取得真实 hostname，保证另一台电脑访问时
      // 请求的是服务器地址，而不是浏览器所在电脑自己的地址。
      function getPageLocation() {
        try { return new URL(window.parent.location.href); } catch (e) {}
        try { return new URL(document.referrer); } catch (e) {}
        try { return new URL(window.location.href); } catch (e) {}
        return new URL("http://localhost");
      }
      var pageLocation = getPageLocation();
      var base = pageLocation.protocol + "//" + pageLocation.hostname + ":__PORT__";
      document.getElementById("livevid").src = base + "/stream?t=" + Date.now();
      function poll(){
        fetch(base + "/status").then(function(r){return r.json();}).then(function(d){
          var t = "画面 " + (d.fps||0).toFixed(1) + " fps · 累计 " + (d.frames||0) +
                  " 帧 · 识别 " + (d.recog_ms||0).toFixed(0) + " ms/次 · 人脸 " +
                  (d.faces||0) + " · 框 " + (d.preview_fps||0).toFixed(1) +
                  " fps · 陌生截图 " + (d.stranger_saves||0) +
                  " · 画面存档 " + (d.snapshot_saves||0) +
                  " · 播放端 " + (d.clients||0);
          if (d.error) { t += " · " + d.error; }
          document.getElementById("livestat").textContent = t;
        }).catch(function(){
          document.getElementById("livestat").textContent =
            "状态获取失败：" + base + "/status（画面可能仍在播放）";
        });
      }
      poll(); setInterval(poll, 1000);
    </script>
    """.replace("__PORT__", str(int(port))).replace("__IMGSTYLE__", img_style)
    components.html(html, height=frame_h)


def _render_stranger_gallery_live(port):
    """在浏览器端实时刷新当前任务的陌生人截图。"""
    html = """
    <style>
      body { margin: 0; font-family: sans-serif; color: #1b3a1e; }
      .meta { background: #fff4f2; border: 1px solid #f1cec8;
              border-radius: 12px; padding: 11px 14px; margin-bottom: 12px;
              font-size: 14px; }
      .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
              gap: 10px; }
      .card { border: 1px solid #e7d5d1; border-radius: 10px; overflow: hidden;
              background: #fff; }
      .card img { width: 100%; display: block; background: #111; }
      .cap { padding: 7px 9px; font-size: 11px; color: #745f5b;
             line-height: 1.4; word-break: break-all; }
      .empty { grid-column: 1 / -1; padding: 28px; text-align: center;
               color: #766662; background: #fff8f7; border-radius: 12px; }
      @media (max-width: 850px) { .grid { grid-template-columns: repeat(2, 1fr); } }
    </style>
    <div class="meta" id="strangermeta">正在连接当前任务…</div>
    <div class="grid" id="strangergrid"></div>
    <script>
      function getPageLocation() {
        try { return new URL(window.parent.location.href); } catch (e) {}
        try { return new URL(document.referrer); } catch (e) {}
        try { return new URL(window.location.href); } catch (e) {}
        return new URL("http://localhost");
      }
      var pageLocation = getPageLocation();
      var base = pageLocation.protocol + "//" + pageLocation.hostname + ":__PORT__";
      var lastKey = "";

      function render(data) {
        var key = (data.task_id || "") + ":" + (data.count || 0);
        if (key === lastKey) { return; }
        lastKey = key;
        document.getElementById("strangermeta").textContent =
          "任务 " + (data.task_id || "-") + " · 已保存陌生人截图 " +
          (data.count || 0) + " 张（本任务上限：" +
          (data.max_images ? (data.max_images + " 张") : "不限") +
          "） · 自动刷新中";
        var grid = document.getElementById("strangergrid");
        grid.textContent = "";
        var events = (data.events || []).slice().reverse();
        if (!events.length) {
          var empty = document.createElement("div");
          empty.className = "empty";
          empty.textContent = "本次任务暂时没有陌生人截图，保存后会自动显示。";
          grid.appendChild(empty);
          return;
        }
        events.forEach(function(event) {
          var card = document.createElement("div");
          card.className = "card";
          var img = document.createElement("img");
          img.src = base + event.image_url + "&t=" + Date.now();
          img.alt = "陌生人截图";
          card.appendChild(img);
          var cap = document.createElement("div");
          cap.className = "cap";
          cap.textContent = (event.created_at || "") + " · " +
                            (event.filename || "");
          card.appendChild(cap);
          grid.appendChild(card);
        });
      }

      function poll() {
        fetch(base + "/stranger?t=" + Date.now(), {cache: "no-store"})
          .then(function(r) { return r.json(); })
          .then(render)
          .catch(function() {
            document.getElementById("strangermeta").textContent =
              "陌生人截图接口连接失败：" + base + "/stranger";
          });
      }
      poll();
      setInterval(poll, 1000);
    </script>
    """.replace("__PORT__", str(int(port)))
    components.html(html, height=700, scrolling=True)


def _render_task_results_live(port):
    """在浏览器端轮询当前任务结果，新图片保存后自动出现。"""
    html = """
    <style>
      body { margin: 0; font-family: sans-serif; color: #1b3a1e; }
      .task-meta {
        background: #f1f7f2; border: 1px solid #d7e8d9; border-radius: 12px;
        padding: 12px 16px; margin-bottom: 14px; font-size: 14px;
      }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 14px; }
      .card { border: 1px solid #d7e8d9; border-radius: 12px; overflow: hidden;
              background: #fff; }
      .card img { width: 100%; display: block; background: #111; }
      .body { padding: 10px 13px 13px; }
      .head { display: flex; justify-content: space-between; gap: 8px;
              align-items: center; margin-bottom: 7px; }
      .badge { color: #fff; border-radius: 999px; padding: 3px 9px;
               font-size: 12px; font-weight: 700; }
      .known { background: #2e7d32; }
      .stranger { background: #c62828; }
      .mixed { background: #ef6c00; }
      .time { color: #607565; font-size: 12px; }
      .file { color: #7a8b7d; font-size: 11px; margin: 2px 0 9px; }
      .result { font-size: 14px; line-height: 1.5; margin: 3px 0; }
      .empty { padding: 30px; text-align: center; color: #667a69;
               background: #f8fbf8; border-radius: 12px; }
      @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
    </style>
    <div class="task-meta" id="taskmeta">正在连接识别任务…</div>
    <div class="grid" id="taskgrid"></div>
    <script>
      function getPageLocation() {
        try { return new URL(window.parent.location.href); } catch (e) {}
        try { return new URL(document.referrer); } catch (e) {}
        try { return new URL(window.location.href); } catch (e) {}
        return new URL("http://localhost");
      }
      var pageLocation = getPageLocation();
      var base = pageLocation.protocol + "//" + pageLocation.hostname + ":__PORT__";
      var labels = {
        known: "数据库人员",
        stranger: "陌生人",
        mixed: "数据库人员 + 陌生人"
      };
      var lastKey = "";

      function text(tag, cls, value) {
        var node = document.createElement(tag);
        if (cls) { node.className = cls; }
        node.textContent = value;
        return node;
      }

      function render(data) {
        var key = (data.task_id || "") + ":" + (data.count || 0);
        if (key === lastKey) { return; }
        lastKey = key;
        var meta = document.getElementById("taskmeta");
        meta.textContent = "任务 " + (data.task_id || "-") +
          " · 已保存结果 " + (data.count || 0) +
          " · 任务开始 " + (data.created_at || "-") +
          " · 自动刷新中";

        var grid = document.getElementById("taskgrid");
        grid.textContent = "";
        var events = (data.events || []).slice().reverse();
        if (!events.length) {
          grid.appendChild(text("div", "empty",
            "当前任务还没有已确认结果；识别结果保存后会自动显示。"));
          return;
        }
        events.forEach(function(event) {
          var card = document.createElement("div");
          card.className = "card";
          var img = document.createElement("img");
          img.src = base + event.image_url + "&t=" + Date.now();
          img.alt = "识别结果图片";
          card.appendChild(img);

          var body = document.createElement("div");
          body.className = "body";
          var head = document.createElement("div");
          head.className = "head";
          var badge = text("span", "badge " + event.category,
                           labels[event.category] || "识别结果");
          head.appendChild(badge);
          head.appendChild(text("span", "time", event.created_at || ""));
          body.appendChild(head);
          body.appendChild(text("div", "file", event.filename || ""));

          (event.results || []).forEach(function(result, i) {
            var line;
            if (result.is_stranger) {
              line = "🚫 人脸 " + (i + 1) + "：陌生人 · 相似度 " +
                     Number(result.score || 0).toFixed(3);
            } else {
              line = "✅ 人脸 " + (i + 1) + "：数据库人员 " +
                     (result.name || "") + " · 相似度 " +
                     Number(result.score || 0).toFixed(3);
            }
            body.appendChild(text("div", "result", line));
          });
          card.appendChild(body);
          grid.appendChild(card);
        });
      }

      function poll() {
        fetch(base + "/task?t=" + Date.now(), {cache: "no-store"})
          .then(function(r) { return r.json(); })
          .then(render)
          .catch(function() {
            document.getElementById("taskmeta").textContent =
              "结果接口连接失败：" + base + "/task";
          });
      }
      poll();
      setInterval(poll, 1000);
    </script>
    """.replace("__PORT__", str(int(port)))
    components.html(html, height=1000, scrolling=True)


def _connect_source(mode, kwargs):
    """建立相机连接，失败时给出提示并返回 None。"""
    try:
        return _get_live_source(mode, kwargs)
    except Exception as exc:  # noqa: BLE001
        banner_err("无法连接相机：%s" % exc)
        if mode == "ros" and not camera.ros_core_ready():
            st.info("ROS master 未运行，请先启动相机驱动：")
            st.code("roscore\n"
                    "roslaunch realsense2_camera rs_camera.launch align_depth:=true",
                    language="bash")
        elif mode == "realsense":
            st.info("请确认 D435i 已插好 USB3 接口，并已执行 "
                    "`pip3 install pyrealsense2`。")
        return None


def page_live():
    _page_header("📹", "实时识别 · Intel RealSense D435i",
                 "完整模式与 face_recognition.launch 的识别节点同流程："
                 "YOLO 人体检测 → InsightFace → 质量过滤 → 多帧投票")

    db = load_db()
    if not db:
        st.warning("人脸库为空，实时识别无法比对身份，请先到「➕ 人脸录入」添加人脸。")

    # ---------- 连接检查 ----------
    with st.expander("🔌 连接检查", expanded=False):
        ros_ok = camera.ros_core_ready()
        c1, c2 = st.columns(2)
        with c1:
            st.write("ROS master：" + ("✅ 在线" if ros_ok
                                      else "❌ 未连接（需先运行 roscore）"))
            st.write("pyrealsense2：" +
                     ("✅ 已安装" if camera.pyrealsense_available()
                      else "❌ 未安装（`pip3 install pyrealsense2`）"))
        with c2:
            topics = camera.list_ros_topics()
            st.write("ROS 话题数：%d" % len(topics))
            image_topics = [t for t in topics
                            if isinstance(t, str)
                            and t.endswith(("/image_raw", "/image_rect_raw"))]
            if image_topics:
                st.write("可用图像话题（可直接填入下方参数）：")
                st.code("\n".join(image_topics), language=None)
        if st.button("🔄 重新检查"):
            st.rerun()
        st.divider()
        st.caption("🚀 相机驱动（roslaunch）：可直接在这里启动/停止，"
                   "无需另开终端。")
        render_launch_panel(st, key_prefix="live")

    # ---------- 取流方式 ----------
    mode_label = st.radio(
        "取流方式",
        ["ROS 话题（配合 realsense2_camera，推荐）",
         "RealSense 直连（pyrealsense2，无需 ROS）"],
        horizontal=True,
    )
    mode = "ros" if mode_label.startswith("ROS") else "realsense"

    kwargs = {}
    with st.expander("取流参数", expanded=False):
        if mode == "ros":
            color_topic = st.text_input("彩色图像话题", "/camera/color/image_raw")
            depth_topic = st.text_input(
                "深度图像话题（留空则不显示深度）",
                "/camera/aligned_depth_to_color/image_raw")
            kwargs.update(
                color_topic=color_topic,
                depth_topic=depth_topic or None,
                depth_max_range=st.slider("深度显示范围(米)", 1.0, 10.0, 4.0, 0.5))
        else:
            res = st.selectbox("分辨率", ["640x480", "848x480", "1280x720"], index=0)
            w, h = [int(v) for v in res.split("x")]
            kwargs.update(
                width=w, height=h,
                fps=st.selectbox("帧率", [15, 30, 60], index=1),
                enable_depth=st.checkbox("启用深度流", value=True),
                depth_max_range=st.slider("深度显示范围(米)", 1.0, 10.0, 4.0, 0.5))

    # ---------- 识别模式 ----------
    rec_mode = st.radio(
        "识别模式",
        ["⚡ 快速（仅人脸检测）",
         "🎯 完整（YOLO+质量过滤+投票，与 ROS 节点一致）"],
        horizontal=True,
        help="快速模式每帧直接检测，延迟低；完整模式与 face_recognition.launch "
             "的识别节点完全一致，更稳但需多帧投票才判定。",
    )
    rec_full = rec_mode.startswith("🎯")
    params = {}

    if rec_full:
        p = recognizer.load_params()
        with st.expander("识别参数（默认来自 config/params.yaml）", expanded=True):
            c1, c2, c3 = st.columns(3)
            params["threshold"] = c1.slider("识别阈值", 0.0, 1.0,
                                            float(p["threshold"]), 0.01)
            params["use_yolo"] = c2.checkbox("启用 YOLO 人体检测",
                                             value=bool(p["use_yolo"]))
            params["enable_voting"] = c3.checkbox("启用多帧投票",
                                                  value=bool(p["enable_voting"]))
            c1, c2, c3 = st.columns(3)
            params["person_confidence"] = c1.slider("人体检测置信度", 0.0, 1.0,
                                                    float(p["person_confidence"]), 0.05)
            params["enable_quality_check"] = c2.checkbox(
                "启用质量过滤", value=bool(p["enable_quality_check"]))
            params["min_det_score"] = c3.slider("最小人脸检测分", 0.0, 1.0,
                                                float(p["min_det_score"]), 0.05)
            c1, c2, c3 = st.columns(3)
            params["vote_window"] = c1.slider("投票窗口(帧)", 5, 60,
                                              int(p["vote_window"]), 1)
            params["vote_min_count"] = c2.slider("最少票数", 1, 30,
                                                 int(p["vote_min_count"]), 1)
            params["vote_min_score"] = c3.slider("计票最低相似度", 0.0, 1.0,
                                                 float(p["vote_min_score"]), 0.05)
            params["track_max_dist"] = st.slider("同一目标最大位移(像素)", 20, 400,
                                                 int(p["track_max_dist"]), 10)
    else:
        with st.expander("识别参数", expanded=True):
            c1, c2 = st.columns(2)
            params["threshold"] = c1.slider("识别阈值", 0.0, 1.0,
                                            float(config.THRESHOLD), 0.01)
            min_det = c2.slider("最小人脸检测分", 0.0, 1.0, 0.3, 0.05)

    # ---------- 视频流参数 ----------
    with st.expander("视频流参数", expanded=False):
        c1, c2, c3 = st.columns(3)
        stream_port = int(c1.number_input("视频流端口", 1024, 65535, 8502, 1))
        jpeg_quality = int(c2.slider("画面质量(JPEG)", 30, 95, 70, 5))
        max_fps = int(c3.slider("画面最高帧率", 5, 60, 15, 5,
                                help="调高画面更流畅，但会占用 CPU 让识别变慢；"
                                     "本机 2 核，建议 15"))
        show_depth = st.checkbox("画面同时显示深度（左右拼接）", value=False,
                                 help="开启后画面宽度减半以容纳深度图")
        video_size = _preview_size_selector(st, key="live_preview_size",
                                            label="预览尺寸")

    # ---------- 控制 ----------
    streaming = live_stream.is_running()
    ctrl = st.columns([1, 1, 1, 1], gap="medium")
    start_btn = ctrl[0].button("▶️ 开始实时视频", type="primary",
                               disabled=streaming, use_container_width=True)
    stop_btn = ctrl[1].button("⏹ 停止视频", disabled=not streaming,
                              use_container_width=True)
    snap = ctrl[2].button("📸 抓拍一帧", disabled=streaming,
                          use_container_width=True)
    reset_btn = ctrl[3].button("♻️ 重置投票", disabled=not rec_full,
                               use_container_width=True)

    if stop_btn:
        live_stream.stop_pipeline()
        _close_live_source()
        _close_recognizer()
        streaming = False
        banner_ok("已停止实时视频流。")

    if reset_btn:
        rec = st.session_state.get("live_rec")
        if rec is not None:
            rec.reset()
        banner_ok("已清空跟踪与投票历史。")

    if start_btn:
        if live_stream.is_running():
            live_stream.stop_pipeline()
        try:
            src = camera.create_source(mode, **kwargs)
        except Exception as exc:  # noqa: BLE001
            banner_err("无法连接相机：%s" % exc)
            if mode == "ros" and not camera.ros_core_ready():
                st.info("ROS master 未运行，请先启动相机驱动：")
                st.code("roscore\nroslaunch realsense2_camera "
                        "rs_camera.launch align_depth:=true", language="bash")
            elif mode == "realsense":
                st.info("请确认 D435i 已插好 USB3 接口，并已执行 "
                        "`pip3 install pyrealsense2`。")
            src = None

        if src is not None:
            base = recognizer.load_params()
            font = recognizer.get_font(base.get("font_path"),
                                       base.get("font_size", 22))
            stranger_saver = _new_stranger_task()
            snapshot_saver = _new_snapshot_task("实时识别")
            analyze, draw = None, None
            if rec_full:
                try:
                    rec = _get_recognizer(repr(("full", sorted(params.items()))),
                                          params, stranger_saver)
                    rec.reset()
                    analyze, draw = rec.analyze, rec.draw
                except Exception as exc:  # noqa: BLE001
                    banner_err("加载完整识别流水线失败：%s" % exc)
            else:
                db_local = load_db()
                analyze = (lambda f: _fast_analyze(f, db_local,
                                                   params["threshold"], min_det,
                                                   stranger_saver))
                draw = (lambda f, infos: recognizer.draw_infos(f, infos, font))

            if analyze is not None:
                try:
                    live_stream.start_pipeline(
                        src, analyze, draw, port=stream_port,
                        jpeg_quality=jpeg_quality, max_fps=max_fps,
                        show_depth=show_depth,
                        snapshot_saver=snapshot_saver,
                        stranger_saver=stranger_saver,
                        preview_detect=recognizer.detect_preview_infos)
                    streaming = True
                    banner_ok("实时视频流已启动（端口 %d）。" % stream_port)
                except Exception as exc:  # noqa: BLE001
                    banner_err("启动视频流失败：%s" % exc)

    with st.expander("🕵️ 本次任务陌生人截图", expanded=False):
        _render_stranger_gallery()
    with st.expander("🖼️ 本次任务识别画面存档", expanded=False):
        _render_recognition_gallery()

    # ========================================================
    # 显示区
    # ========================================================
    if streaming:
        pipe = live_stream.get_pipeline()
        _render_mjpeg(pipe.server.port, size=video_size)
        st.caption("画面为连续视频流；人脸框通过快速检测和光流跟随，"
                   "姓名/陌生人结果约每秒更新一次。"
                   "如看不到画面，请确认端口 %d 可访问。" % pipe.server.port)
        return

    # ========================================================
    # 抓拍一帧
    # ========================================================
    if not snap:
        st.info("点击「▶️ 开始实时视频」即可看到连续画面与人脸框，"
                "或点击「📸 抓拍一帧」抓取单张画面。")
        return

    src = _connect_source(mode, kwargs)
    if src is None:
        return
    snapshot_path, snapshot_category = None, None
    snap_stranger_saver = _new_stranger_task()
    with st.spinner("正在抓取并识别..."):
        color, depth, _ = src.read(timeout=3.0)
        if color is None:
            display, infos = None, []
        elif rec_full:
            try:
                rec = _get_recognizer(repr(("full", sorted(params.items()))),
                                      params, snap_stranger_saver)
                display, infos = rec.process(color)
            except Exception as exc:  # noqa: BLE001
                banner_err("完整识别失败：%s" % exc)
                display, infos = color, []
        else:
            display, infos = _fast_process(
                color, db, params["threshold"], min_det,
                stranger_saver=snap_stranger_saver,
            )
        if display is not None and infos:
            task = _new_snapshot_task("单帧抓拍")
            snapshot_path, snapshot_category = task.save(display, infos)
    if color is None:
        banner_warn("未收到图像数据：请检查话题名或相机是否已启动。")
        return
    _render_frame(display, depth)
    if rec_full:
        st.caption("完整模式需要多帧投票(默认 ≥8 票)才判定身份，"
                   "单次抓拍通常显示为“识别中…”，连续监测或多次抓拍后会给出结果。")
    if snapshot_path:
        st.caption("识别画面已保存（%s）：%s"
                   % (fc.RECOGNITION_SNAPSHOT_LABELS[snapshot_category],
                      snapshot_path))
    _render_live_results(infos)


# ============================================================
# 主流程
# ============================================================

def main():
    _init_state()
    _inject_css()
    # 打开网页就自动拉起 roslaunch（可在侧边栏关闭）
    maybe_autolaunch()
    msg = st.session_state.get("launch_msg", "")
    if msg.startswith("err:") and st.session_state.get("launch_auto"):
        banner_warn("自动启动 roslaunch 失败：" + msg[4:])
    page = render_sidebar()

    ok, err = ensure_model()
    if not ok:
        banner_err("模型加载失败：%s" % err)
        st.stop()

    if page == "📊 概览":
        page_overview()
    elif page == "➕ 人脸录入":
        page_enroll()
    elif page == "🗂️ 人脸库管理":
        page_manage()
    elif page == "📹 实时识别":
        page_live()
    elif page == "🖼️ 本次任务结果":
        page_task_results()
    elif page == "🔍 识别测试":
        page_test()


if __name__ == "__main__":
    main()
