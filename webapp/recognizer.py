#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与 ROS 节点 scripts/face_recognition_node.py **一致** 的人脸识别流水线，
供网页端实时监测使用：

    YOLO 人体检测 -> 人体区域内 InsightFace 人脸检测/特征
    -> 质量过滤 -> 多帧跟踪 + 投票 -> 输出

参数默认读取 config/params.yaml，与节点保持完全一致（可在网页里临时调整）。
"""

import os
from collections import Counter, deque

import cv2
import numpy as np

import config
import face_core as fc


# 与 config/params.yaml 对应的默认值
DEFAULTS = {
    "use_yolo": True,
    "yolo_model": "yolo11n.pt",
    "person_confidence": 0.35,
    "threshold": 0.4,
    "font_path": "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "font_size": 22,
    # 方法一：质量过滤
    "enable_quality_check": True,
    "min_det_score": 0.60,
    "min_eye_ratio": 0.25,
    "min_mouth_nose_ratio": 0.15,
    "enable_106_check": False,
    "min_mouth_std": 1.5,
    # 方法二：多帧投票
    "enable_voting": True,
    "vote_window": 15,
    "vote_min_count": 8,
    "vote_min_score": 0.35,
    "track_max_dist": 120,
    "track_max_age": 30,
}


def load_params(path=None):
    """读取 ROS 节点的 params.yaml，缺失项用默认值补齐。"""
    params = dict(DEFAULTS)
    path = path or config.PARAMS_FILE
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key in DEFAULTS:
            if key in data and data[key] is not None:
                params[key] = data[key]
    except Exception:
        pass
    return params


def _resolve_yolo(path):
    """YOLO 权重可能是相对路径，逐个候选目录找。"""
    if path and os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        os.path.join(config.PKG_DIR, path or ""),
        os.path.join(config.PKG_DIR, "src", "InsightFace", path or ""),
        os.path.join(os.getcwd(), path or ""),
    ]
    for c in candidates:
        if path and os.path.exists(c):
            return c
    return path


def _load_font(path, size):
    try:
        from PIL import ImageFont
        return ImageFont.truetype(path, int(size))
    except Exception:
        return None


_font_cache = {}


def get_font(path=None, size=22):
    """加载并缓存中文字体（供网页/视频流绘制标签使用）。"""
    path = path or DEFAULTS["font_path"]
    key = (str(path), int(size))
    if key not in _font_cache:
        _font_cache[key] = _load_font(path, size)
    return _font_cache[key]


def draw_infos(frame, infos, font=None, font_size=22):
    """
    把识别结果画到给定帧上（原地修改并返回）。

    infos 每项需含 bbox / status(或 name+score) / color，
    这样采集线程可以把“最近一次识别结果”叠加到每一帧新画面上。
    """
    if not infos:
        return frame
    labels = []
    for info in infos:
        x1, y1, x2, y2 = [int(round(v)) for v in info["bbox"]]
        color = tuple(info.get("color", (0, 255, 0)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = info.get("status") or ("%s %.2f"
                                      % (info.get("name", "?"),
                                         info.get("score", 0.0)))
        labels.append((text, (x1, max(0, y1 - font_size - 6)), color))
    return _draw_labels(frame, labels, font)


def _draw_labels(img, labels, font):
    if not labels:
        return img
    if font is None:
        return _draw_labels_cv2(img, labels)
    from PIL import Image as PILImage, ImageDraw
    pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, pos, color_bgr in labels:
        draw.text(pos, text, font=font,
                  fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_labels_cv2(img, labels):
    for text, pos, color_bgr in labels:
        cv2.putText(img, text.encode("ascii", "ignore").decode() or "?",
                    pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    return img


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detect_preview_infos(frame):
    """
    快速人脸框预检测（不提取特征、不做身份识别）。

    SCRFD 使用较小的 320x320 输入，在 CPU 上约 0.1 秒；供采集线程高频
    更新人脸框位置。正式身份结果返回后会沿用这些框的位置和姓名/陌生人
    状态。
    """
    app = fc.get_analyzer()
    bboxes, _ = app.det_model.detect(
        frame, input_size=(320, 320), max_num=0, metric="default"
    )
    if bboxes is None:
        return []
    h, w = frame.shape[:2]
    infos = []
    for row in bboxes:
        x1, y1, x2, y2, score = [float(v) for v in row[:5]]
        x1 = max(0, min(w, int(round(x1))))
        y1 = max(0, min(h, int(round(y1))))
        x2 = max(0, min(w, int(round(x2))))
        y2 = max(0, min(h, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        infos.append({
            "bbox": (x1, y1, x2, y2),
            "name": "__PENDING__",
            "score": float(score),
            "quality": "pending",
            "status": "识别中...",
            "color": (0, 255, 255),
        })
    infos.sort(
        key=lambda d: (d["bbox"][2] - d["bbox"][0])
        * (d["bbox"][3] - d["bbox"][1]),
        reverse=True,
    )
    return infos


class LiveRecognizer(object):
    """带 YOLO 人体检测、质量过滤与多帧投票的识别器（与 ROS 节点同逻辑）。"""

    def __init__(self, params=None, db=None, stranger_saver=None):
        self.params = dict(params) if params else load_params()
        self.db = db if db is not None else fc.load_db()
        self.stranger_saver = (
            stranger_saver if stranger_saver is not None
            else fc.StrangerFaceSaver()
        )
        self.app = fc.get_analyzer()

        self.yolo = None
        if self.params.get("use_yolo", True):
            from ultralytics import YOLO
            self.yolo = YOLO(_resolve_yolo(self.params.get("yolo_model",
                                                            "yolo11n.pt")))

        self.font = _load_font(self.params.get("font_path"),
                               self.params.get("font_size", 22))
        self.reset()

    # --------------------------------------------------------
    # 状态
    # --------------------------------------------------------

    def reset(self):
        """清空跟踪与投票历史。"""
        self.tracks = []
        self.frame_id = 0

    def update_db(self, db):
        self.db = db

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def set_stranger_saver(self, stranger_saver):
        self.stranger_saver = stranger_saver

    # --------------------------------------------------------
    # 人脸收集（YOLO 人体检测 + InsightFace）
    # --------------------------------------------------------

    def collect_faces(self, frame):
        """返回 [((gx1, gy1, gx2, gy2), face), ...]。"""
        items = []
        h, w = frame.shape[:2]

        if self.yolo is not None:
            results = self.yolo(frame,
                                conf=self.params.get("person_confidence", 0.35),
                                verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0].item()) != 0:      # COCO: person = 0
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                for face in self.app.get(person_img):
                    bx1, by1, bx2, by2 = face.bbox.astype(int)
                    items.append(((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                                  face))
        else:
            for face in self.app.get(frame):
                bx1, by1, bx2, by2 = face.bbox.astype(int)
                items.append(((bx1, by1, bx2, by2), face))
        return items

    # --------------------------------------------------------
    # 识别（返回原始最佳，不阈值化）—— 与节点 _recognize_face_raw 一致
    # --------------------------------------------------------

    def recognize_raw(self, embedding):
        q = fc.l2_normalize(embedding)
        best_name, best_score = "Unknown", 0.0
        for person_name, data in self.db.items():
            template = data.get("template")
            if template is not None:
                s = float(np.dot(q, fc.l2_normalize(template)))
                if s > best_score:
                    best_score, best_name = s, person_name
            for e in data.get("embeddings", []):
                s = float(np.dot(q, fc.l2_normalize(e)))
                if s > best_score:
                    best_score, best_name = s, person_name
        return best_name, best_score

    # --------------------------------------------------------
    # 质量过滤 —— 与节点 _face_quality_check 一致
    # --------------------------------------------------------

    def quality_check(self, face):
        p = self.params
        if face.det_score < p.get("min_det_score", 0.60):
            return False, "检测分低%.2f" % face.det_score

        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 5:
            return True, "ok"

        x1, y1, x2, y2 = face.bbox
        face_w = max(x2 - x1, 1e-6)
        face_h = max(y2 - y1, 1e-6)

        eye_dist = np.linalg.norm(kps[0] - kps[1])
        if eye_dist / face_w < p.get("min_eye_ratio", 0.25):
            return False, "侧脸/低头"

        eye_mid_y = (kps[0][1] + kps[1][1]) / 2.0
        if kps[2][1] <= eye_mid_y:
            return False, "关键点错乱"

        mouth_nose = (np.linalg.norm(kps[3] - kps[2]) +
                      np.linalg.norm(kps[4] - kps[2])) / 2.0
        if mouth_nose / face_h < p.get("min_mouth_nose_ratio", 0.15):
            return False, "疑似捂嘴"

        if p.get("enable_106_check", False):
            lm = getattr(face, "landmark_2d_106", None)
            if lm is not None:
                mouth_std = float(lm[96:106].std(axis=0).mean())
                if mouth_std < p.get("min_mouth_std", 1.5):
                    return False, "嘴部塌缩"
        return True, "ok"

    # --------------------------------------------------------
    # 跟踪 + 投票 —— 与节点一致
    # --------------------------------------------------------

    def _find_or_create_track(self, center, cur_frame_id):
        best_i, best_d = -1, self.params.get("track_max_dist", 120)
        for i, t in enumerate(self.tracks):
            cx, cy = t["center"]
            d = ((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            self.tracks[best_i]["center"] = center
            self.tracks[best_i]["last_seen"] = cur_frame_id
            return self.tracks[best_i]
        t = {"center": center,
             "history": deque(maxlen=self.params.get("vote_window", 15)),
             "last_seen": cur_frame_id}
        self.tracks.append(t)
        return t

    def _vote_from_history(self, history):
        votes = Counter()
        scores_by_name = {}
        for name, score in history:
            if name == "__BAD__":
                continue
            # Unknown 代表“最高相似度仍低于身份阈值”，不是低质量结果。
            # 它也应该参与投票，否则低相似度的陌生人永远无法确认并变红。
            if (name != "Unknown"
                    and score < self.params.get("vote_min_score", 0.35)):
                continue
            votes[name] += 1
            scores_by_name.setdefault(name, []).append(score)
        if not votes:
            return "Unknown", 0, 0.0
        best_name, best_count = votes.most_common(1)[0]
        return best_name, best_count, float(np.mean(scores_by_name[best_name]))

    # --------------------------------------------------------
    # 单张人脸处理 —— 与节点 _process_one_face 一致
    # --------------------------------------------------------

    def _process_one_face(self, bbox, face, frame=None):
        gx1, gy1, gx2, gy2 = bbox
        embedding = np.asarray(face.embedding, dtype=np.float32)

        track = self._find_or_create_track(_bbox_center(bbox), self.frame_id)
        history = track["history"]

        if self.params.get("enable_quality_check", True):
            ok, reason = self.quality_check(face)
        else:
            ok, reason = True, "ok"

        if ok:
            raw_name, raw_score = self.recognize_raw(embedding)
            if raw_score >= self.params.get("threshold", 0.4):
                single_name, single_score = raw_name, raw_score
            else:
                single_name, single_score = "Unknown", raw_score
            # 投票应使用阈值化后的结果，这样低相似度的人会被当作
            # Unknown 投票，而不是一直停留在“识别中”。
            history.append((single_name, single_score))
        else:
            raw_name, raw_score = "__BAD__", 0.0
            single_name, single_score = "__BAD__", 0.0
            history.append(("__BAD__", 0.0))

        voted_name, voted_count, voted_score = self._vote_from_history(history)
        voted_ok = (self.params.get("enable_voting", True)
                    and voted_count >= self.params.get("vote_min_count", 8))

        if voted_ok:
            final_name, final_score = voted_name, voted_score
        elif not ok:
            final_name, final_score = "__BAD__", 0.0
        else:
            final_name, final_score = single_name, single_score

        is_stranger = final_name == "Unknown"
        result_ready = bool(
            voted_ok or not self.params.get("enable_voting", True)
        )
        stranger_confirmed = bool(
            is_stranger and result_ready
        )
        saved_path = None
        if (stranger_confirmed and ok and frame is not None
                and self.stranger_saver is not None):
            saved_path = self.stranger_saver.save(frame, bbox, embedding)

        if is_stranger:
            status = "陌生人 %.3f" % final_score
            if saved_path:
                status += "（已保存）"
            color = (0, 0, 255)
        elif voted_ok:
            status = "%s %.3f" % (final_name, final_score)
            color = (0, 255, 0)
        elif not ok:
            status = "! %s" % reason
            color = (0, 165, 255)
        else:
            status = "%s %.3f" % (single_name, single_score)
            color = (0, 255, 255)

        return {
            "bbox": (int(gx1), int(gy1), int(gx2), int(gy2)),
            "name": final_name,
            "score": float(final_score),
            "raw_score": float(raw_score),
            "quality": reason if not ok else "ok",
            "vote_count": int(voted_count),
            "confirmed": bool(voted_ok),
            "is_stranger": bool(is_stranger),
            "stranger_confirmed": bool(stranger_confirmed),
            "result_ready": result_ready,
            "stranger_saved": saved_path,
            "status": status,
            "color": color,
        }

    # --------------------------------------------------------
    # 绘制（中文标签用 PIL，无中文字体时退回英文）
    # --------------------------------------------------------

    def draw(self, frame, infos):
        """把识别结果画到给定帧上（原地修改并返回）。"""
        return draw_infos(frame, infos, self.font,
                          self.params.get("font_size", 22))

    # --------------------------------------------------------
    # 主入口：只做识别分析，不画图
    # --------------------------------------------------------

    def analyze(self, frame):
        """
        处理一帧 BGR 图像，返回 infos（人脸几何 + 姓名 + 状态）。

        供实时视频流使用：画面由采集线程绘制，这里只负责“算”。
        """
        self.frame_id += 1
        infos = [self._process_one_face(bbox, face, frame)
                 for bbox, face in self.collect_faces(frame)]

        # 清理过期轨迹（与节点一致）
        max_age = self.params.get("track_max_age", 30)
        self.tracks = [t for t in self.tracks
                       if self.frame_id - t["last_seen"] < max_age]
        return infos

    # --------------------------------------------------------
    # 兼容旧用法：识别 + 画图
    # --------------------------------------------------------

    def process(self, frame):
        """返回 (display_bgr, infos)。"""
        infos = self.analyze(frame)
        return self.draw(frame.copy(), infos), infos
