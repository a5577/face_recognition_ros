#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别网页应用 —— 核心逻辑。

功能:
    - InsightFace(buffalo_l) 模型加载与缓存
    - 人脸检测 + 512 维特征提取（L2 归一化）
    - 人脸特征数据库的读取 / 写入 / 增删改
    - 余弦相似度识别（与 ROS 节点 scripts/face_recognition_node.py 逻辑一致）

数据库格式（与 ROS 节点兼容）::

    {
        "姓名": {
            "embeddings": [np.ndarray(512), ...],   # 每张照片一条特征
            "template":   np.ndarray(512),          # 平均模板（已归一化）
            "images":     {"1.jpg": np.ndarray(512), ...},  # 照片名 -> 特征（可选）
        },
        ...
    }
"""

import os
import re
import pickle
import threading
import time

import cv2
import numpy as np

import config


# ============================================================
# 模型
# ============================================================

_analyzer = None
_analyzer_lock = threading.Lock()


def get_analyzer():
    """
    加载（并缓存）InsightFace 模型。首次调用约需数秒。

    注意：本机内存较紧张，全程只保留一个实例（约 1GB），
    因此固定使用 config.DET_SIZE，与 ROS 节点保持一致。
    """
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name=config.MODEL_NAME,
                    root=config.MODEL_ROOT,
                    providers=config.PROVIDERS,
                )
                # ctx_id<0 表示强制使用 CPU
                ctx_id = 0 if any("CUDA" in p for p in config.PROVIDERS) else -1
                app.prepare(ctx_id=ctx_id, det_size=config.DET_SIZE)
                _analyzer = app
    return _analyzer


# ============================================================
# 特征工具
# ============================================================

def l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return vec
    return (vec / n).astype(np.float32)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recompute_template(embeddings):
    """由多条特征计算平均模板并归一化。"""
    if not embeddings:
        return None
    mat = np.stack([l2_normalize(e) for e in embeddings], axis=0)
    return l2_normalize(mat.mean(axis=0))


# ============================================================
# 人脸检测 / 特征提取
# ============================================================

def detect_faces(bgr, sort_by_area=True):
    """
    对一张 BGR 图片做人脸检测 + 特征提取。

    返回列表，每项::

        {
            "bbox": [x1, y1, x2, y2],
            "det_score": float,
            "embedding": np.ndarray(512)  # 已 L2 归一化
        }

    默认按人脸面积从大到小排序，第 0 个即“主脸”。
    """
    app = get_analyzer()
    faces = app.get(bgr)
    results = []
    for f in faces:
        emb = l2_normalize(f.embedding)
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        results.append(
            {
                "bbox": [x1, y1, x2, y2],
                "det_score": float(f.det_score),
                "embedding": emb,
            }
        )
    if sort_by_area:
        results.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0])
            * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
    return results


def draw_faces(bgr, faces, color=(0, 255, 0), thickness=2, label=True):
    """在图片副本上绘制人脸框，用于网页预览。"""
    img = bgr.copy()
    for f in faces:
        x1, y1, x2, y2 = [int(round(v)) for v in f["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            txt = "%.2f" % f["det_score"]
            cv2.putText(
                img, txt, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
    return img


# ============================================================
# 图片编解码（兼容中文路径）
# ============================================================

def decode_image(data_bytes):
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imread(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path, bgr):
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".jpg"
        path = path + ext
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        return False
    buf.tofile(path)
    return True


# ============================================================
# 陌生人截图
# ============================================================

_stranger_file_lock = threading.Lock()


def crop_face(bgr, bbox, margin=None):
    """
    按人脸框裁出人脸，并向外扩展少量上下文。

    返回 BGR 裁图；框无效或图片为空时返回 None。
    """
    if bgr is None or bbox is None:
        return None
    if margin is None:
        margin = config.STRANGER_CROP_MARGIN

    h, w = bgr.shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    dx, dy = bw * float(margin), bh * float(margin)
    x1 = max(0, min(w, int(round(x1 - dx))))
    y1 = max(0, min(h, int(round(y1 - dy))))
    x2 = max(0, min(w, int(round(x2 + dx))))
    y2 = max(0, min(h, int(round(y2 + dy))))
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2].copy()


def save_stranger_face(bgr, bbox, directory=None):
    """
    将陌生人脸裁图保存为 JPEG。

    返回保存路径；保存失败时返回 None。
    """
    face = crop_face(bgr, bbox)
    if face is None or face.size == 0:
        return None

    directory = directory or config.STRANGER_DIR
    try:
        with _stranger_file_lock:
            os.makedirs(directory, exist_ok=True)
            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            stamp += "_%06d" % int((now % 1.0) * 1000000)
            path = os.path.join(directory, "stranger_%s.jpg" % stamp)
            seq = 2
            while os.path.exists(path):
                path = os.path.join(
                    directory, "stranger_%s_%02d.jpg" % (stamp, seq)
                )
                seq += 1
            return path if imwrite(path, face) else None
    except OSError:
        return None


class StrangerFaceSaver(object):
    """
    带人脸特征去重的陌生人截图器。

    默认保存每个不同陌生人各一张，同一张脸不会重复保存；也可通过
    ``max_images`` 设置当前任务的硬上限。该类线程安全，可直接放进
    实时识别线程。
    """

    def __init__(self, directory=None, similarity=None, task_id=None,
                 max_images=None):
        self.directory = directory or config.STRANGER_DIR
        self.task_id = task_id or time.strftime("stranger_task_%Y%m%d_%H%M%S")
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.similarity = float(
            config.STRANGER_SIMILARITY if similarity is None else similarity
        )
        self.max_images = max(0, int(
            config.STRANGER_MAX_PER_TASK
            if max_images is None else max_images
        ))
        self._recent = []
        self._events = []
        self._lock = threading.Lock()

    def _is_duplicate(self, embedding):
        q = l2_normalize(embedding)
        if float(np.linalg.norm(q)) < 1e-12:
            return False
        return any(
            cosine_similarity(q, old) >= self.similarity
            for old in self._recent
        )

    def save(self, bgr, bbox, embedding):
        """
        保存一张陌生人脸。

        返回新截图路径；重复人脸、无效裁图或写入失败时返回 None。
        """
        with self._lock:
            if (self.max_images > 0
                    and len(self._events) >= self.max_images):
                return None
            if self._is_duplicate(embedding):
                return None
            path = save_stranger_face(bgr, bbox, self.directory)
            if path is not None:
                self._recent.append(l2_normalize(embedding))
                self._events.append({
                    "path": path,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            return path

    def get_events(self):
        """返回当前任务的陌生人截图事件。"""
        with self._lock:
            return [dict(event) for event in self._events]

    def get_event(self, index):
        """按序号取得当前任务的一张陌生人截图。"""
        with self._lock:
            try:
                return dict(self._events[int(index)])
            except (IndexError, TypeError, ValueError):
                return None

    def get_task_data(self):
        """返回供实时陌生人截图面板轮询的数据。"""
        with self._lock:
            events = [
                {
                    "filename": os.path.basename(event["path"]),
                    "created_at": event["created_at"],
                }
                for event in self._events
            ]
            return {
                "task_id": self.task_id,
                "created_at": self.created_at,
                "max_images": self.max_images,
                "count": len(events),
                "events": events,
            }


def list_stranger_images(directory=None, limit=None):
    """按修改时间倒序列出陌生人截图路径。"""
    directory = directory or config.STRANGER_DIR
    if not os.path.isdir(directory):
        return []
    paths = []
    for name in os.listdir(directory):
        if not name.lower().endswith(config.IMAGE_EXTENSIONS):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            paths.append(path)
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if limit is not None:
        return paths[:int(limit)]
    return paths


# ============================================================
# 识别画面存档（数据库人员 / 陌生人 / 混合）
# ============================================================

_snapshot_file_lock = threading.Lock()
_snapshot_font_cache = {}
RECOGNITION_SNAPSHOT_CATEGORIES = ("known", "stranger", "mixed")
RECOGNITION_SNAPSHOT_LABELS = {
    "known": "数据库人员",
    "stranger": "陌生人",
    "mixed": "数据库人员 + 陌生人",
}


def classify_recognition(infos):
    """
    按本帧识别结果分类。

    返回 ``{category, names, stranger_count, results}``；没有可保存的
    有效人脸结果时返回 None。数据库人员使用 ``known``，陌生人为
    ``stranger``，同一帧同时出现两类时为 ``mixed``。
    """
    names = set()
    stranger_count = 0
    results = []
    for info in infos or []:
        if info.get("bbox") is None:
            continue
        if not info.get("result_ready", True):
            continue
        quality = info.get("quality", "ok")
        if quality not in (None, "ok"):
            continue
        name = str(info.get("name", "Unknown"))
        if name in ("", "__BAD__"):
            continue
        is_stranger = bool(
            name == "Unknown" or info.get("is_stranger", False)
        )
        if is_stranger:
            stranger_count += 1
        else:
            names.add(name)
        results.append({
            "name": name,
            "score": float(info.get("score", 0.0)),
            "is_stranger": is_stranger,
            "status": str(info.get("status", name)),
        })

    if names and stranger_count:
        category = "mixed"
    elif names:
        category = "known"
    elif stranger_count:
        category = "stranger"
    else:
        return None
    return {
        "category": category,
        "names": sorted(names),
        "stranger_count": stranger_count,
        "results": results,
    }


def _get_snapshot_font(size):
    from PIL import ImageFont
    key = (config.RECOGNITION_SNAPSHOT_FONT_PATH, int(size))
    if key not in _snapshot_font_cache:
        try:
            _snapshot_font_cache[key] = ImageFont.truetype(
                config.RECOGNITION_SNAPSHOT_FONT_PATH, int(size)
            )
        except Exception:
            _snapshot_font_cache[key] = ImageFont.load_default()
    return _snapshot_font_cache[key]


def _draw_snapshot_banner(bgr, event):
    """在存档画面底部加上“数据库人员/陌生人”类型条。"""
    img = bgr.copy()
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return img

    names = event["names"]
    shown = "、".join(names[:4])
    if len(names) > 4:
        shown += "等 %d 人" % len(names)

    category = event["category"]
    if category == "known":
        text = "数据库人员：%s" % shown
        color = (0, 180, 0)
    elif category == "stranger":
        text = "陌生人：%d 人" % event["stranger_count"]
        color = (0, 0, 255)
    else:
        text = "数据库人员：%s  |  陌生人：%d 人" % (
            shown, event["stranger_count"]
        )
        color = (0, 165, 255)

    bar_h = max(38, min(64, int(round(h * 0.085))))
    y0 = max(0, h - bar_h)
    cv2.rectangle(img, (0, y0), (w, h), (15, 15, 15), -1)
    cv2.rectangle(img, (0, y0), (max(7, int(w * 0.012)), h), color, -1)

    try:
        from PIL import Image as PILImage, ImageDraw
        font = _get_snapshot_font(max(16, min(26, bar_h - 16)))
        pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text(
            (max(16, int(w * 0.02)), y0 + max(8, (bar_h - 24) // 2)),
            text,
            font=font,
            fill=(255, 255, 255),
        )
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        fallback = {
            "known": "DATABASE",
            "stranger": "STRANGER",
            "mixed": "DATABASE + STRANGER",
        }[category]
        cv2.putText(
            img, fallback, (16, y0 + bar_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )
        return img


def save_recognition_snapshot(bgr, infos, directory=None):
    """
    保存一张带人脸框和数据库/陌生人类型条的完整识别画面。

    返回 ``(path, category)``；没有有效人脸或写入失败时返回 ``(None, None)``。
    """
    event = classify_recognition(infos)
    if bgr is None or event is None:
        return None, None

    category = event["category"]
    directory = directory or config.RECOGNITION_SNAPSHOT_DIR
    target_dir = os.path.join(directory, category)
    img = _draw_snapshot_banner(bgr, event)
    try:
        with _snapshot_file_lock:
            os.makedirs(target_dir, exist_ok=True)
            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            stamp += "_%06d" % int((now % 1.0) * 1000000)
            path = os.path.join(
                target_dir, "%s_%s.jpg" % (category, stamp)
            )
            seq = 2
            while os.path.exists(path):
                path = os.path.join(
                    target_dir,
                    "%s_%s_%02d.jpg" % (category, stamp, seq),
                )
                seq += 1
            return (path, category) if imwrite(path, img) else (None, None)
    except OSError:
        return None, None


class RecognitionFrameSaver(object):
    """一个识别任务内的画面存档器，同一识别结果只保存一次。"""

    def __init__(self, directory=None, task_id=None):
        self.directory = directory or config.RECOGNITION_SNAPSHOT_DIR
        self.task_id = task_id or time.strftime("task_%Y%m%d_%H%M%S")
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._saved_signatures = set()
        self._events = []
        self._lock = threading.Lock()

    @property
    def event_count(self):
        with self._lock:
            return len(self._events)

    def get_events(self):
        """返回当前任务已保存的结果快照（按保存顺序）。"""
        with self._lock:
            return [
                dict(event, results=[dict(item) for item in event["results"]])
                for event in self._events
            ]

    def get_event(self, index):
        """按序号取得当前任务的一条结果；越界时返回 None。"""
        with self._lock:
            try:
                event = self._events[int(index)]
            except (IndexError, TypeError, ValueError):
                return None
            return dict(
                event,
                results=[dict(item) for item in event["results"]],
            )

    def get_task_data(self):
        """返回供任务结果页面实时轮询的数据。"""
        with self._lock:
            events = [
                {
                    "category": event["category"],
                    "created_at": event["created_at"],
                    "filename": os.path.basename(event["path"]),
                    "results": [dict(item) for item in event["results"]],
                }
                for event in self._events
            ]
            return {
                "task_id": self.task_id,
                "created_at": self.created_at,
                "count": len(events),
                "events": events,
            }

    def save(self, bgr, infos):
        """
        保存当前任务中的一个新识别结果。

        同一“类别 + 姓名组合 + 陌生人数”在任务内只保存一次。返回
        ``(path, category)``；重复结果、结果尚未确认、无有效人脸或写入
        失败时返回 ``(None, None)``。
        """
        event = classify_recognition(infos)
        if bgr is None or event is None:
            return None, None
        signature = (
            event["category"],
            tuple(event["names"]),
            event["stranger_count"],
        )
        with self._lock:
            if signature in self._saved_signatures:
                return None, None
            result = save_recognition_snapshot(
                bgr, infos, self.directory
            )
            if result[0] is not None:
                self._saved_signatures.add(signature)
                event.update({
                    "path": result[0],
                    "category": result[1],
                    "task_id": self.task_id,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                self._events.append(event)
            return result


def list_recognition_snapshots(directory=None, category=None, limit=None):
    """按修改时间倒序列出识别画面存档路径。"""
    directory = directory or config.RECOGNITION_SNAPSHOT_DIR
    categories = (
        [category] if category in RECOGNITION_SNAPSHOT_CATEGORIES
        else list(RECOGNITION_SNAPSHOT_CATEGORIES)
    )
    paths = []
    for kind in categories:
        target = os.path.join(directory, kind)
        if not os.path.isdir(target):
            continue
        for name in os.listdir(target):
            if not name.lower().endswith(config.IMAGE_EXTENSIONS):
                continue
            path = os.path.join(target, name)
            if os.path.isfile(path):
                paths.append(path)
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if limit is not None:
        return paths[:int(limit)]
    return paths


def recognition_snapshot_category(path):
    """从存档路径取得类别；无法识别时返回空字符串。"""
    kind = os.path.basename(os.path.dirname(path))
    return kind if kind in RECOGNITION_SNAPSHOT_CATEGORIES else ""


# ============================================================
# 数据库读写
# ============================================================

def _normalize_entry(data):
    """把任意格式的一条记录统一成 {embeddings, template, images}。"""
    embeddings = []
    images = {}
    template = None

    if isinstance(data, dict):
        for e in data.get("embeddings", []) or []:
            embeddings.append(l2_normalize(e))
        raw_imgs = data.get("images", {}) or {}
        if isinstance(raw_imgs, dict):
            for k, v in raw_imgs.items():
                images[str(k)] = l2_normalize(v)
        tpl = data.get("template", None)
        if tpl is not None:
            template = l2_normalize(tpl)
    elif isinstance(data, (list, tuple)):
        # 兼容 build_database.py 的 {name: [emb, emb, ...]} 格式
        for e in data:
            embeddings.append(l2_normalize(e))

    if not embeddings and images:
        embeddings = list(images.values())
    if template is None and embeddings:
        template = recompute_template(embeddings)
    return {"embeddings": embeddings, "template": template, "images": images}


def _compat_unpickle(path):
    """兼容 np>=2 生成的旧文件（numpy._core -> numpy.core）。"""
    import sys
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_db(path=None):
    """读取数据库，返回统一格式的 dict。文件不存在时返回空 dict。"""
    path = path or config.DB_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        raw = _compat_unpickle(path)
    db = {}
    for name, data in raw.items():
        db[str(name)] = _normalize_entry(data)
    return db


def save_db(db, path=None):
    """写入数据库（pickle），同时保证 template 与 embeddings 一致。"""
    path = path or config.DB_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for name, entry in db.items():
        embs = [l2_normalize(e) for e in entry.get("embeddings", [])]
        images = {
            str(k): l2_normalize(v)
            for k, v in (entry.get("images") or {}).items()
        }
        tpl = entry.get("template", None)
        tpl = l2_normalize(tpl) if tpl is not None else recompute_template(embs)
        out[name] = {"embeddings": embs, "template": tpl, "images": images}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return path


# ============================================================
# 识别
# ============================================================

def recognize_face(query_embedding, db, threshold=None):
    """
    在数据库中查找最相似的身份。

    返回 (name, score, confirmed)：
        name      —— 命中的身份名，未超过阈值时为 "Unknown"
        score     —— 最高相似度
        confirmed —— score >= threshold
    """
    if threshold is None:
        threshold = config.THRESHOLD
    q = l2_normalize(query_embedding)

    best_name = "Unknown"
    best_score = -1.0
    for person_name, data in db.items():
        template = data.get("template")
        if template is not None:
            s = float(np.dot(q, template))
            if s > best_score:
                best_score, best_name = s, person_name
        for e in data.get("embeddings", []):
            s = float(np.dot(q, l2_normalize(e)))
            if s > best_score:
                best_score, best_name = s, person_name

    if best_score >= threshold:
        return best_name, best_score, True
    return "Unknown", max(best_score, 0.0), False


# ============================================================
# 照片文件管理
# ============================================================

def person_dir(name):
    return os.path.join(config.FACE_DATA_DIR, name)


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_person_images(name):
    d = person_dir(name)
    if not os.path.isdir(d):
        return []
    files = [
        f for f in os.listdir(d)
        if f.lower().endswith(config.IMAGE_EXTENSIONS)
    ]
    files.sort(key=_natural_key)
    return files


def next_image_name(name, ext=".jpg"):
    """生成不与现有文件冲突的文件名，如 13.jpg。"""
    d = person_dir(name)
    max_idx = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            stem = os.path.splitext(f)[0]
            if stem.isdigit():
                max_idx = max(max_idx, int(stem))
    return "%d%s" % (max_idx + 1, ext)


def list_people():
    """列出 face_data 下的所有身份（目录名）。"""
    if not os.path.isdir(config.FACE_DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(config.FACE_DATA_DIR)
        if os.path.isdir(person_dir(d))
    )


# ============================================================
# 高层操作：录入 / 删除 / 重建
# ============================================================

def enroll_images(name, image_list, require_single=False, min_det_score=0.0):
    """
    将若干张图片录入到指定身份下（身份不存在则新建）。

    参数:
        name           身份名称
        image_list     [(文件名, BGR 图片), ...]
        require_single 为 True 时，一张图检测到多张人脸则跳过
        min_det_score  小于该检测分的图片跳过

    返回 (db, results)，results 每项::

        {"file": str, "status": "ok"/"skip", "reason": str,
         "saved_as": str, "det_score": float, "n_faces": int}
    """
    db = load_db()
    entry = db.get(name, {"embeddings": [], "template": None, "images": {}})
    os.makedirs(person_dir(name), exist_ok=True)

    results = []
    for fname, bgr in image_list:
        if bgr is None:
            results.append({"file": fname, "status": "skip",
                            "reason": "图片读取失败", "n_faces": 0})
            continue

        faces = detect_faces(bgr)  # 已按面积降序
        n = len(faces)
        if n == 0:
            results.append({"file": fname, "status": "skip",
                            "reason": "未检测到人脸", "n_faces": 0})
            continue
        if require_single and n > 1:
            results.append({"file": fname, "status": "skip",
                            "reason": "检测到 %d 张人脸（要求单张）" % n,
                            "n_faces": n})
            continue

        best = faces[0]
        if best["det_score"] < min_det_score:
            results.append({"file": fname, "status": "skip",
                            "reason": "检测分 %.2f 低于阈值 %.2f"
                                      % (best["det_score"], min_det_score),
                            "n_faces": n})
            continue

        saved_as = next_image_name(name)
        save_path = os.path.join(person_dir(name), saved_as)
        if not imwrite(save_path, bgr):
            results.append({"file": fname, "status": "skip",
                            "reason": "文件保存失败", "n_faces": n})
            continue

        entry["images"][saved_as] = best["embedding"]
        entry["embeddings"].append(best["embedding"])
        results.append({"file": fname, "status": "ok", "reason": "",
                        "saved_as": saved_as,
                        "bbox": best["bbox"],
                        "det_score": best["det_score"], "n_faces": n})

    entry["template"] = recompute_template(entry["embeddings"])
    if entry["embeddings"]:
        db[name] = entry
    save_db(db)
    return db, results


def rebuild_person(name, db=None, _save=True):
    """
    按 face_data/<name> 目录下的照片重新计算全部特征。
    用于修复数据，或删除单张照片后无法对应特征时。
    """
    db = db if db is not None else load_db()
    files = list_person_images(name)
    images = {}
    for fname in files:
        bgr = imread(os.path.join(person_dir(name), fname))
        if bgr is None:
            continue
        faces = detect_faces(bgr)
        if not faces:
            continue
        images[fname] = faces[0]["embedding"]
    if not images:
        entry = None
        db.pop(name, None)
    else:
        entry = {
            "embeddings": list(images.values()),
            "template": recompute_template(list(images.values())),
            "images": images,
        }
        db[name] = entry
    if _save:
        save_db(db)
    return entry


def delete_image(name, filename, db=None):
    """删除某身份下的一张照片，并同步更新特征。"""
    db = db if db is not None else load_db()
    path = os.path.join(person_dir(name), filename)
    if os.path.exists(path):
        os.remove(path)

    entry = db.get(name)
    if entry is not None:
        entry.get("images", {}).pop(filename, None)
        if entry.get("images"):
            entry["embeddings"] = list(entry["images"].values())
            entry["template"] = recompute_template(entry["embeddings"])
        else:
            # 记录与文件名无法一一对应时，按剩余照片重建
            entry = rebuild_person(name, db=db, _save=False)

    if entry is None or not entry.get("embeddings"):
        db.pop(name, None)
        d = person_dir(name)
        if os.path.isdir(d) and not list_person_images(name):
            try:
                os.rmdir(d)
            except OSError:
                pass
    else:
        db[name] = entry
    save_db(db)
    return db


def delete_person(name, db=None, remove_files=True):
    """删除整个身份（照片 + 特征）。"""
    db = db if db is not None else load_db()
    db.pop(name, None)
    if remove_files:
        d = person_dir(name)
        if os.path.isdir(d):
            for f in list_person_images(name):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass
            try:
                os.rmdir(d)
            except OSError:
                pass
    save_db(db)
    return db


def rename_person(old_name, new_name, db=None):
    """重命名身份（照片目录 + 数据库键）。"""
    db = db if db is not None else load_db()
    if old_name == new_name:
        return db

    old_dir = person_dir(old_name)
    new_dir = person_dir(new_name)
    if os.path.isdir(old_dir):
        if os.path.exists(new_dir):
            raise FileExistsError("已存在同名身份：%s" % new_name)
        os.rename(old_dir, new_dir)

    entry = db.pop(old_name, None)
    if entry is not None:
        db[new_name] = entry
    save_db(db)
    return db


def rebuild_all(db=None):
    """按脸库目录重建所有人的特征。返回 (db, stats)。"""
    db = db if db is not None else load_db()
    stats = {}
    people = set(list_people()) | set(db.keys())
    for name in sorted(people):
        entry = rebuild_person(name, db=db, _save=False)
        stats[name] = 0 if entry is None else len(entry["embeddings"])
    save_db(db)
    return db, stats


# ============================================================
# 统计
# ============================================================

def db_stats(db):
    n_people = len(db)
    n_emb = sum(len(e.get("embeddings", [])) for e in db.values())
    return n_people, n_emb
