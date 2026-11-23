#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时视频流（MJPEG over HTTP）—— 让网页显示连续画面，而不是一帧一帧刷新。

设计要点：把“画面帧率”和“识别帧率”解耦

    采集线程：按相机帧率抓帧 -> 光流移动人脸框 -> JPEG 编码 -> 推给 MJPEG 服务
    预检测线程：SCRFD 小尺寸快速检测框 -> 让人脸框快速出现并按约 4 fps 校正
    识别线程：按自身速度（CPU 约 1~1.5 fps）跑识别流水线，更新人脸框与姓名
    网页    ：用 <img src="http://host:port/stream"> 播放，画面和框位置流畅，
              姓名/陌生人身份约每秒更新一次

只用标准库 http.server，无需额外依赖。
"""

import atexit
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np


# ============================================================
# MJPEG HTTP 服务
# ============================================================

class _StreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):        # 静默日志
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def do_GET(self):
        srv = self.server.stream_server
        path = self.path.split("?")[0]

        if path == "/status":
            body = json.dumps(srv.get_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/task":
            body = json.dumps(
                srv.get_task_data(), ensure_ascii=False
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/task-image":
            query = parse_qs(urlsplit(self.path).query)
            index = (query.get("i") or ["-1"])[0]
            body = srv.get_task_image(index)
            if body is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self._cors()
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/stranger":
            body = json.dumps(
                srv.get_stranger_data(), ensure_ascii=False
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/stranger-image":
            query = parse_qs(urlsplit(self.path).query)
            index = (query.get("i") or ["-1"])[0]
            body = srv.get_stranger_image(index)
            if body is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self._cors()
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path != "/stream":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()
            return

        # ---- MJPEG 流 ----
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self._cors()
        self.end_headers()

        srv.add_client()
        last = -1
        try:
            while not srv.stopped:
                with srv.cond:
                    while srv.seq == last and not srv.stopped:
                        srv.cond.wait(timeout=1.0)
                    last = srv.seq
                    jpg = srv.latest_jpeg
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    ("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            srv.remove_client()


class MjpegServer(object):
    """极简 MJPEG 服务：后台线程推送最新 JPEG 帧。"""

    def __init__(self, port=8502):
        self.port = int(port)
        self.latest_jpeg = None
        self.seq = 0
        self.stopped = False
        self.clients = 0
        self.cond = threading.Condition()
        self._httpd = None
        self._thread = None
        self._status = {}
        self.task_saver = None
        self.stranger_saver = None

    def start(self):
        if self._httpd is not None:
            return self
        self.stopped = False
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _StreamHandler)
        self._httpd.stream_server = self
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self.stopped = True
        with self.cond:
            self.cond.notify_all()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

    def publish(self, jpeg_bytes):
        with self.cond:
            self.latest_jpeg = jpeg_bytes
            self.seq += 1
            self.cond.notify_all()

    def add_client(self):
        with self.cond:
            self.clients += 1

    def remove_client(self):
        with self.cond:
            self.clients = max(0, self.clients - 1)

    def set_status(self, **kwargs):
        with self.cond:
            self._status.update(kwargs)

    def get_status(self):
        with self.cond:
            st = dict(self._status)
            st["clients"] = self.clients
            return st

    def set_task_saver(self, task_saver):
        with self.cond:
            self.task_saver = task_saver

    def get_task_data(self):
        with self.cond:
            saver = self.task_saver
        if saver is None:
            return {"count": 0, "events": []}
        data = saver.get_task_data()
        for i, event in enumerate(data["events"]):
            event["image_url"] = "/task-image?i=%d" % i
        return data

    def get_task_image(self, index):
        with self.cond:
            saver = self.task_saver
        if saver is None:
            return None
        event = saver.get_event(index)
        if event is None:
            return None
        try:
            with open(event["path"], "rb") as f:
                return f.read()
        except (KeyError, OSError):
            return None

    def set_stranger_saver(self, stranger_saver):
        with self.cond:
            self.stranger_saver = stranger_saver

    def get_stranger_data(self):
        with self.cond:
            saver = self.stranger_saver
        if saver is None:
            return {"count": 0, "events": []}
        data = saver.get_task_data()
        for i, event in enumerate(data["events"]):
            event["image_url"] = "/stranger-image?i=%d" % i
        return data

    def get_stranger_image(self, index):
        with self.cond:
            saver = self.stranger_saver
        if saver is None:
            return None
        event = saver.get_event(index)
        if event is None:
            return None
        try:
            with open(event["path"], "rb") as f:
                return f.read()
        except (KeyError, OSError):
            return None


# ============================================================
# 采集 + 识别流水线
# ============================================================

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _merge_preview_infos(preview_infos, result_infos):
    """
    用快速检测框的位置，承载正式识别结果的姓名/陌生人状态。

    正式识别约每秒更新一次；快速检测约 5～8 次/秒。两者按 IoU 或中心距离
    匹配后，显示框取快速检测的位置，身份信息取正式识别结果。
    """
    if not preview_infos:
        return [dict(info) for info in result_infos]
    if not result_infos:
        return [dict(info) for info in preview_infos]

    used = set()
    merged = []
    for preview in preview_infos:
        pbox = preview["bbox"]
        pcx = (pbox[0] + pbox[2]) / 2.0
        pcy = (pbox[1] + pbox[3]) / 2.0
        best_i, best_score = -1, 0.0
        for i, result in enumerate(result_infos):
            if i in used:
                continue
            rbox = result["bbox"]
            iou = _bbox_iou(pbox, rbox)
            rcx = (rbox[0] + rbox[2]) / 2.0
            rcy = (rbox[1] + rbox[3]) / 2.0
            size = max(
                pbox[2] - pbox[0], pbox[3] - pbox[1],
                rbox[2] - rbox[0], rbox[3] - rbox[1], 1.0,
            )
            dist = ((pcx - rcx) ** 2 + (pcy - rcy) ** 2) ** 0.5
            score = iou if iou >= 0.15 else max(0.0, 0.9 - dist / size)
            if score > best_score:
                best_i, best_score = i, score
        if best_i >= 0 and best_score > 0.0:
            item = dict(result_infos[best_i])
            item["bbox"] = pbox
            merged.append(item)
            used.add(best_i)
        else:
            merged.append(dict(preview))

    for i, result in enumerate(result_infos):
        if i not in used:
            merged.append(dict(result))
    return merged


class _OpticalBoxTracker(object):
    """用稀疏光流在快速检测之间平滑移动人脸框。"""

    def __init__(self):
        self.prev_gray = None
        self.states = []

    @staticmethod
    def _features(gray, bbox):
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        points = cv2.goodFeaturesToTrack(
            gray[y1:y2, x1:x2],
            maxCorners=30, qualityLevel=0.01, minDistance=4,
        )
        if points is None:
            return None
        points = points.reshape(-1, 1, 2)
        points[:, 0, 0] += x1
        points[:, 0, 1] += y1
        return points.astype(np.float32)

    def update(self, frame, infos, reset=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (reset or self.prev_gray is None
                or self.prev_gray.shape != gray.shape
                or len(infos) != len(self.states)):
            self.states = [{
                "info": dict(info),
                "bbox": tuple(info["bbox"]),
                "points": self._features(gray, info["bbox"]),
            } for info in infos]
        else:
            h, w = gray.shape[:2]
            for state, info in zip(self.states, infos):
                state["info"] = dict(info)
                points = state.get("points")
                if points is not None and len(points) >= 3:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, points, None,
                        winSize=(21, 21), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS
                                  | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                    )
                    if moved is not None and status is not None:
                        good = status.reshape(-1) == 1
                        old = points.reshape(-1, 2)[good]
                        new = moved.reshape(-1, 2)[good]
                        if len(old) >= 3:
                            delta = np.median(new - old, axis=0)
                            x1, y1, x2, y2 = state["bbox"]
                            dx, dy = float(delta[0]), float(delta[1])
                            moved_box = (
                                max(0, min(w - 1, int(round(x1 + dx)))),
                                max(0, min(h - 1, int(round(y1 + dy)))),
                                max(0, min(w, int(round(x2 + dx)))),
                                max(0, min(h, int(round(y2 + dy)))),
                            )
                            if (moved_box[2] > moved_box[0]
                                    and moved_box[3] > moved_box[1]):
                                state["bbox"] = moved_box
                            state["points"] = new.reshape(-1, 1, 2)
                if state["points"] is None:
                    state["points"] = self._features(gray, state["bbox"])

        self.prev_gray = gray
        return [
            dict(state["info"], bbox=state["bbox"])
            for state in self.states
        ]


class LivePipeline(object):
    """
    后台采集线程 + 识别线程。

    source  : camera.FrameSource（read() 返回 color, depth, stamp）
    analyze : callable(frame_bgr) -> infos                识别（返回人脸几何/姓名）
    draw    : callable(frame_bgr, infos) -> frame_bgr     把结果画到当前帧
    snapshot_saver : 可选，保存带标注的完整识别画面
    preview_detect : 可选，低成本返回快速人脸框，用于高频跟随
    """

    def __init__(self, source, analyze, draw, server, jpeg_quality=70,
                 max_fps=25, show_depth=False, depth_max_range=4.0,
                 snapshot_saver=None, preview_detect=None,
                 preview_max_fps=4, stranger_saver=None):
        self.source = source
        self.analyze = analyze
        self.draw = draw
        self.snapshot_saver = snapshot_saver
        self.stranger_saver = stranger_saver
        self.preview_detect = preview_detect
        self.server = server
        self.jpeg_quality = int(jpeg_quality)
        self.show_depth = show_depth
        self.depth_max_range = depth_max_range
        self.interval = 1.0 / float(max(1, max_fps))
        self.preview_interval = 1.0 / float(max(1, preview_max_fps))

        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._lock = threading.Lock()
        self._raw = None
        self._raw_seq = 0
        self._infos = []
        self._preview_infos = []
        self._preview_seq = 0
        self._display_preview_seq = -1
        self._box_tracker = _OpticalBoxTracker()
        self._threads = []
        self._stats = {"fps": 0.0, "frames": 0, "faces": 0,
                       "stranger_saves": 0, "snapshot_saves": 0,
                       "recog_ms": 0.0, "preview_fps": 0.0,
                       "error": "", "running": False}

    def start(self):
        self.server.start()
        self._stop.clear()
        self._stats["running"] = True
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
        ]
        # analyze 为 None 时只做预览，不启动识别线程（省 CPU）
        if self.analyze is not None:
            self._threads.append(
                threading.Thread(target=self._recognize_loop, daemon=True))
        if self.analyze is not None and self.preview_detect is not None:
            self._threads.append(
                threading.Thread(target=self._preview_loop, daemon=True))
        for t in self._threads:
            t.start()
        return self

    def grab(self):
        """取当前最新一帧原始画面（用于抓拍）。没有则返回 None。"""
        with self._cond:
            return None if self._raw is None else self._raw.copy()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        self._stats["running"] = False
        try:
            self.source.close()
        except Exception:
            pass
        self.server.stop()

    @property
    def running(self):
        return (not self._stop.is_set()) and bool(self._threads)

    def get_state(self):
        with self._lock:
            st = dict(self._stats)
        st["clients"] = self.server.clients
        return st

    # ---------- 采集线程 ----------

    def _capture_loop(self):
        count = 0
        t0 = time.time()
        while not self._stop.is_set():
            loop_t0 = time.time()
            color, depth, _ = self.source.read(timeout=1.0)
            if color is None:
                self.server.set_status(fps=0.0, error="等待相机数据")
                time.sleep(0.1)
                continue

            with self._cond:                 # 交给识别线程
                self._raw = color.copy()
                self._raw_seq += 1
                self._cond.notify_all()

            with self._lock:                 # 叠加最近一次识别结果
                infos = _merge_preview_infos(
                    list(self._preview_infos), list(self._infos)
                )
                preview_seq = self._preview_seq
            infos = self._box_tracker.update(
                color, infos, reset=preview_seq != self._display_preview_seq
            )
            self._display_preview_seq = preview_seq
            frame = self.draw(color, infos) if (infos and self.draw) else color
            if self.show_depth and depth is not None:
                h = min(frame.shape[0], depth.shape[0])
                frame = np.hstack([frame[:h], depth[:h]])

            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                self.server.publish(buf.tobytes())

            count += 1
            now = time.time()
            fps = count / max(now - t0, 1e-6)
            with self._lock:
                self._stats["frames"] = count
                self._stats["fps"] = fps
            self.server.set_status(fps=fps, frames=count)

            dt = time.time() - loop_t0
            if dt < self.interval:
                time.sleep(self.interval - dt)

    # ---------- 快速人脸框检测线程 ----------

    def _preview_loop(self):
        last = -1
        count = 0
        t0 = time.time()
        while not self._stop.is_set():
            loop_t0 = time.time()
            with self._cond:
                while self._raw_seq == last and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                last = self._raw_seq
                frame = self._raw
            if frame is None:
                continue
            try:
                infos = self.preview_detect(frame)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._preview_infos = []
                    self._preview_seq += 1
                self.server.set_status(
                    preview_error="%s: %s" % (type(exc).__name__, exc)
                )
                time.sleep(0.2)
                continue

            count += 1
            now = time.time()
            preview_fps = count / max(now - t0, 1e-6)
            with self._lock:
                self._preview_infos = infos
                self._preview_seq += 1
                self._stats["preview_fps"] = preview_fps
            self.server.set_status(
                preview_fps=preview_fps, preview_faces=len(infos),
                preview_error="",
            )
            dt = time.time() - loop_t0
            if dt < self.preview_interval:
                time.sleep(self.preview_interval - dt)

    # ---------- 识别线程 ----------

    def _recognize_loop(self):
        last = -1
        while not self._stop.is_set():
            with self._cond:
                while self._raw_seq == last and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                last = self._raw_seq
                frame = self._raw
            if frame is None:
                continue

            t = time.time()
            try:
                infos = self.analyze(frame)
                ms = (time.time() - t) * 1000.0
                saved = sum(1 for info in infos
                            if info.get("stranger_saved"))
                snapshot_saved = False
                if self.snapshot_saver is not None and self.draw and infos:
                    try:
                        display = self.draw(frame.copy(), infos)
                        snapshot_path, _ = self.snapshot_saver.save(
                            display, infos
                        )
                        snapshot_saved = snapshot_path is not None
                    except Exception:  # 截图失败不能中断实时识别
                        snapshot_saved = False
                with self._lock:
                    self._infos = infos
                    self._stats["faces"] = len(infos)
                    self._stats["stranger_saves"] += saved
                    self._stats["snapshot_saves"] += int(snapshot_saved)
                    stranger_saves = self._stats["stranger_saves"]
                    snapshot_saves = self._stats["snapshot_saves"]
                    self._stats["recog_ms"] = ms
                    self._stats["error"] = ""
                self.server.set_status(
                    faces=len(infos), recog_ms=ms,
                    stranger_saves=stranger_saves,
                    snapshot_saves=snapshot_saves, error="",
                )
            except Exception as exc:  # noqa: BLE001
                msg = "%s: %s" % (type(exc).__name__, exc)
                with self._lock:
                    self._stats["error"] = msg
                self.server.set_status(error=msg)
                time.sleep(0.5)


# ============================================================
# 全局单例（跨 Streamlit rerun / 浏览器刷新都保持运行）
# ============================================================

_server = None
_pipeline = None
# 可重入锁：start_pipeline 持锁期间会调用 get_server，后者也需要拿这把锁
_lock = threading.RLock()


def get_server(port=8502):
    global _server
    with _lock:
        if _server is None or _server.port != int(port) or _server.stopped:
            if _server is not None:
                try:
                    _server.stop()
                except Exception:
                    pass
            _server = MjpegServer(port).start()
        return _server


def start_pipeline(source, analyze, draw, port=8502, **kwargs):
    """停掉旧流水线，启动新流水线。"""
    global _pipeline
    with _lock:
        if _pipeline is not None:
            try:
                _pipeline.stop()
            except Exception:
                pass
            _pipeline = None
        server = get_server(port)
        pipeline = LivePipeline(source, analyze, draw, server, **kwargs)
        server.set_task_saver(pipeline.snapshot_saver)
        server.set_stranger_saver(pipeline.stranger_saver)
        _pipeline = pipeline.start()
        return _pipeline


def stop_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is not None:
            try:
                _pipeline.stop()
            except Exception:
                pass
            _pipeline = None


def get_pipeline():
    return _pipeline


def is_running():
    return _pipeline is not None and _pipeline.running


def grab_frame():
    """取当前流水线的最新一帧（用于抓拍）；没有在跑则返回 None。"""
    pipe = _pipeline
    if pipe is None or not pipe.running:
        return None
    return pipe.grab()


def _shutdown():
    """解释器退出时清理线程与端口，避免卡在退出流程。"""
    try:
        stop_pipeline()
    except Exception:
        pass
    global _server
    try:
        if _server is not None:
            _server.stop()
            _server = None
    except Exception:
        pass


atexit.register(_shutdown)
