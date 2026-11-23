#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 人脸识别节点
订阅图像话题，执行 YOLO 人员检测 + InsightFace 识别 + 质量过滤 + 多帧投票，
发布识别结果（FaceResultArray）和可视化图像。
"""

import os
import cv2
import pickle
import time
import numpy as np
from collections import deque, Counter

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from PIL import Image as PILImage, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from ultralytics import YOLO

from face_recognition_ros.msg import FaceResult, FaceResultArray


# ============================================================
# 工具函数
# ============================================================

def feature_to_str(emb, dim=None, decimals=3):
    if emb is None:
        return "None"
    emb = np.asarray(emb, dtype=np.float32)
    if dim is not None:
        arr = emb[:dim]
        s = ", ".join(f"{v:.{decimals}f}" for v in arr)
        return f"[{s}, ...] (norm={np.linalg.norm(emb):.3f})"
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in emb) + "]"


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ============================================================
# 节点类
# ============================================================

class FaceRecognitionNode(object):

    def __init__(self):
        rospy.init_node("face_recognition_node", anonymous=False)

        # ---------------- 读取 ROS 参数 ----------------
        self.database_file    = rospy.get_param("~database_file", "")
        self.use_yolo         = rospy.get_param("~use_yolo", True)
        self.yolo_model       = rospy.get_param("~yolo_model", "yolo11n.pt")
        self.person_conf      = rospy.get_param("~person_confidence", 0.35)
        self.threshold        = rospy.get_param("~threshold", 0.4)

        self.font_path        = rospy.get_param("~font_path",
                                                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
        self.font_size        = rospy.get_param("~font_size", 22)

        self.show_feature_dim = rospy.get_param("~show_feature_dim", 6)
        self.print_full_feat  = rospy.get_param("~print_full_feature", False)
        self.save_feature_log = rospy.get_param("~save_feature_log", True)
        self.feature_log_file = rospy.get_param("~feature_log_file",
                                                "face_feature_log.txt")
        self.print_interval   = rospy.get_param("~print_interval", 1.0)

        # 方法一：质量过滤
        self.enable_quality_check = rospy.get_param("~enable_quality_check", True)
        self.min_det_score        = rospy.get_param("~min_det_score", 0.60)
        self.min_eye_ratio        = rospy.get_param("~min_eye_ratio", 0.25)
        self.min_mouth_nose_ratio = rospy.get_param("~min_mouth_nose_ratio", 0.15)
        self.enable_106_check     = rospy.get_param("~enable_106_check", False)
        self.min_mouth_std        = rospy.get_param("~min_mouth_std", 1.5)

        # 方法二：多帧投票
        self.enable_voting    = rospy.get_param("~enable_voting", True)
        self.vote_window      = rospy.get_param("~vote_window", 15)
        self.vote_min_count   = rospy.get_param("~vote_min_count", 8)
        self.vote_min_score   = rospy.get_param("~vote_min_score", 0.35)
        self.track_max_dist   = rospy.get_param("~track_max_dist", 120)
        self.track_max_age    = rospy.get_param("~track_max_age", 30)

        # 话题
        self.image_topic       = rospy.get_param("~image_topic", "/camera/image_raw")
        self.result_topic      = rospy.get_param("~result_topic",
                                                 "/face_recognition/results")
        self.vis_topic         = rospy.get_param("~visualization_topic",
                                                 "/face_recognition/visualization")
        self.publish_vis       = rospy.get_param("~publish_visualization", True)

        # ---------------- 中文字体 ----------------
        try:
            self.cn_font = ImageFont.truetype(self.font_path, self.font_size)
        except Exception as e:
            rospy.logwarn("加载字体失败 %s，回退到默认字体: %s" % (self.font_path, e))
            self.cn_font = ImageFont.load_default()

        # ---------------- 加载 InsightFace ----------------
        rospy.loginfo("正在加载 InsightFace...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        rospy.loginfo("InsightFace 加载完成")

        # ---------------- 加载 YOLO ----------------
        self.yolo = None
        if self.use_yolo:
            rospy.loginfo("正在加载 YOLO: %s ..." % self.yolo_model)
            self.yolo = YOLO(self.yolo_model)
            rospy.loginfo("YOLO 加载完成")

        # ---------------- 加载人脸数据库 ----------------
        rospy.loginfo("正在加载人脸数据库: %s" % self.database_file)
        if not self.database_file or not os.path.exists(self.database_file):
            raise FileNotFoundError("找不到数据库文件: %s" % self.database_file)

        with open(self.database_file, "rb") as f:
            self.face_database = pickle.load(f)

        rospy.loginfo("数据库人数: %d" % len(self.face_database))
        for name, data in self.face_database.items():
            if isinstance(data, dict):
                n = len(data.get("embeddings", []))
                rospy.loginfo("  - %s: %d 条特征 + 平均模板" % (name, n))
            else:
                rospy.loginfo("  - %s: %d 条特征" % (name, len(data)))

        # ---------------- 日志文件 ----------------
        self.log_fp = None
        if self.save_feature_log:
            log_dir = os.path.dirname(self.feature_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.log_fp = open(self.feature_log_file, "a", encoding="utf-8")
            self.log_fp.write(
                "\n===== 会话开始 %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S")
            )
            self.log_fp.flush()

        # ---------------- 状态 ----------------
        self.tracks             = []
        self.frame_id           = 0
        self.last_print_time    = 0.0
        self.current_frame_info = []

        # ---------------- CvBridge（仅用于订阅端解码）----------------
        self.bridge = CvBridge()

        # ---------------- 发布者 ----------------
        self.result_pub = rospy.Publisher(self.result_topic,
                                          FaceResultArray, queue_size=1)
        self.vis_pub = None
        if self.publish_vis:
            self.vis_pub = rospy.Publisher(self.vis_topic, Image, queue_size=1)

        # ---------------- 订阅者 ----------------
        self.sub = rospy.Subscriber(self.image_topic, Image,
                                    self.image_callback,
                                    queue_size=1, buff_size=2 ** 24)

        rospy.loginfo("人脸识别节点已启动，订阅: %s" % self.image_topic)
        rospy.loginfo("结果发布: %s" % self.result_topic)
        if self.publish_vis:
            rospy.loginfo("可视化图像发布: %s" % self.vis_topic)

        rospy.on_shutdown(self.on_shutdown)

    # ============================================================
    # 回调
    # ============================================================

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("图像转换失败: %s" % e)
            return

        self.frame_id += 1
        display = frame.copy()
        labels = []
        self.current_frame_info = []

        # -------- 1. 收集人脸 --------
        face_items = self._collect_faces(frame)

        # -------- 2. 逐脸处理 --------
        for (gx1, gy1, gx2, gy2), face in face_items:
            self._process_one_face(
                frame, display, labels,
                (gx1, gy1, gx2, gy2), face
            )

        # -------- 3. 清理过期轨迹 --------
        self.tracks = [t for t in self.tracks
                       if self.frame_id - t["last_seen"] < self.track_max_age]

        # -------- 4. 绘制中文标签 --------
        display = self._draw_labels(display, labels)

        # -------- 5. 发布识别结果 --------
        self._publish_results(msg.header)

        # -------- 6. 发布可视化图像（手动构造 sensor_msgs/Image，绕过 cv_bridge）--------
        if self.publish_vis and self.vis_pub is not None:
            try:
                vis_img = display
                if vis_img.dtype != np.uint8:
                    vis_img = vis_img.astype(np.uint8)
                if vis_img.ndim == 3 and vis_img.shape[2] == 3:
                    vis_msg = Image()
                    vis_msg.header = msg.header
                    vis_msg.height = vis_img.shape[0]
                    vis_msg.width = vis_img.shape[1]
                    vis_msg.encoding = "bgr8"
                    vis_msg.is_bigendian = False
                    vis_msg.step = vis_img.shape[1] * 3
                    vis_msg.data = vis_img.tobytes()
                    self.vis_pub.publish(vis_msg)
                else:
                    rospy.logwarn("跳过可视化发布，图像格式不匹配: shape=%s"
                                  % str(vis_img.shape))
            except Exception as e:
                rospy.logerr("可视化图像发布失败: %s" % e)

        # -------- 7. 周期性打印与日志 --------
        self._periodic_log()

    # ============================================================
    # 人脸收集
    # ============================================================

    def _collect_faces(self, frame):
        face_items = []
        h, w = frame.shape[:2]

        if self.use_yolo and self.yolo is not None:
            results = self.yolo(frame, conf=self.person_conf, verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0].item()) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                faces = self.app.get(person_img)
                for face in faces:
                    bx1, by1, bx2, by2 = face.bbox.astype(int)
                    gx1, gy1 = bx1 + x1, by1 + y1
                    gx2, gy2 = bx2 + x1, by2 + y1
                    face_items.append(((gx1, gy1, gx2, gy2), face))
        else:
            faces = self.app.get(frame)
            for face in faces:
                bx1, by1, bx2, by2 = face.bbox.astype(int)
                face_items.append(((bx1, by1, bx2, by2), face))

        return face_items

    # ============================================================
    # 单张人脸处理
    # ============================================================

    def _process_one_face(self, frame, display, labels, bbox, face):
        gx1, gy1, gx2, gy2 = bbox
        embedding = face.embedding

        center = bbox_center(bbox)
        track = self._find_or_create_track(center, self.frame_id)
        history = track["history"]

        # -------- 质量过滤 --------
        if self.enable_quality_check:
            ok, reason = self._face_quality_check(face)
        else:
            ok, reason = True, "ok"

        # -------- 识别 + 写投票历史 --------
        if ok:
            raw_name, raw_score, db_emb = self._recognize_face_raw(embedding)
            history.append((raw_name, raw_score))

            if raw_score >= self.threshold:
                single_name, single_score = raw_name, raw_score
            else:
                single_name, single_score = "Unknown", raw_score
        else:
            raw_name, raw_score, db_emb = "__BAD__", 0.0, None
            single_name, single_score = "__BAD__", 0.0
            history.append(("__BAD__", 0.0))

        # -------- 多帧投票 --------
        voted_name, voted_count, voted_score = self._vote_from_history(history)
        voted_ok = self.enable_voting and voted_count >= self.vote_min_count

        # -------- 显示逻辑 --------
        if voted_ok:
            final_name = voted_name
            final_score = voted_score
            if final_name == "Unknown":
                status = "Unknown %.3f" % final_score
                color = (0, 0, 255)      # 红
            else:
                status = "%s %.3f" % (final_name, final_score)
                color = (0, 255, 0)      # 绿
        elif not ok:
            status = "! %s" % reason
            color = (0, 165, 255)        # 橙
        else:
            if single_name != "Unknown":
                status = "%s %.3f" % (single_name, single_score)
            else:
                status = "识别中... %.3f" % single_score
            color = (0, 255, 255)        # 黄
            final_name = single_name
            final_score = single_score

        # -------- 画框 + 标签 --------
        cv2.rectangle(display, (gx1, gy1), (gx2, gy2), color, 2)
        labels.append((status,
                       (gx1, max(0, gy1 - self.font_size - 6)),
                       color))

        if self.show_feature_dim > 0 and ok:
            feat_str = feature_to_str(embedding, dim=self.show_feature_dim)
            labels.append(("feat%d: %s" % (self.show_feature_dim, feat_str),
                           (gx1, gy2 + 4), color))

        # -------- 缓存 --------
        self.current_frame_info.append({
            "bbox":         (gx1, gy1, gx2, gy2),
            "name":         voted_name if voted_ok else single_name,
            "score":        voted_score if voted_ok else single_score,
            "raw_score":    raw_score,
            "embedding":    np.asarray(embedding, dtype=np.float32),
            "db_embedding": db_emb,
            "quality":      reason if not ok else "ok",
            "vote_count":   voted_count,
            "voted_ok":     voted_ok,
        })

    # ============================================================
    # 识别（返回原始最佳，不阈值化）
    # ============================================================

    def _recognize_face_raw(self, query_embedding):
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-6)

        best_name = "Unknown"
        best_score = -1.0
        best_db_emb = None

        for person_name, data in self.face_database.items():
            if isinstance(data, dict):
                template = data.get("template", None)
                embs = data.get("embeddings", [])
            else:
                template = None
                embs = data

            if template is not None:
                s = float(np.dot(q, template))
                if s > best_score:
                    best_score = s
                    best_name = person_name
                    best_db_emb = template

            for e in embs:
                e = np.asarray(e, dtype=np.float32)
                e = e / max(np.linalg.norm(e), 1e-6)
                s = float(np.dot(q, e))
                if s > best_score:
                    best_score = s
                    best_name = person_name
                    best_db_emb = e

        return best_name, best_score, best_db_emb

    # ============================================================
    # 方法一：质量过滤
    # ============================================================

    def _face_quality_check(self, face):
        if face.det_score < self.min_det_score:
            return False, "检测分低%.2f" % face.det_score

        kps = face.kps
        x1, y1, x2, y2 = face.bbox
        face_w = max(x2 - x1, 1e-6)
        face_h = max(y2 - y1, 1e-6)

        eye_dist = np.linalg.norm(kps[0] - kps[1])
        if eye_dist / face_w < self.min_eye_ratio:
            return False, "侧脸/低头"

        eye_mid_y = (kps[0][1] + kps[1][1]) / 2.0
        if kps[2][1] <= eye_mid_y:
            return False, "关键点错乱"

        mouth_nose = (np.linalg.norm(kps[3] - kps[2]) +
                      np.linalg.norm(kps[4] - kps[2])) / 2.0
        if mouth_nose / face_h < self.min_mouth_nose_ratio:
            return False, "疑似捂嘴"

        if self.enable_106_check:
            lm = getattr(face, "landmark_2d_106", None)
            if lm is not None:
                mouth_pts = lm[96:106]
                mouth_std = float(mouth_pts.std(axis=0).mean())
                if mouth_std < self.min_mouth_std:
                    return False, "嘴部塌缩"

        return True, "ok"

    # ============================================================
    # 方法二：多帧时序投票
    # ============================================================

    def _find_or_create_track(self, center, cur_frame_id):
        best_i = -1
        best_d = self.track_max_dist

        for i, t in enumerate(self.tracks):
            cx, cy = t["center"]
            d = ((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_i = i

        if best_i >= 0:
            self.tracks[best_i]["center"] = center
            self.tracks[best_i]["last_seen"] = cur_frame_id
            return self.tracks[best_i]

        t = {
            "center":    center,
            "history":   deque(maxlen=self.vote_window),
            "last_seen": cur_frame_id,
        }
        self.tracks.append(t)
        return t

    def _vote_from_history(self, history):
        votes = Counter()
        scores_by_name = {}

        for name, score in history:
            if name == "__BAD__":
                continue
            if score < self.vote_min_score:
                continue
            votes[name] += 1
            scores_by_name.setdefault(name, []).append(score)

        if not votes:
            return "Unknown", 0, 0.0

        best_name, best_count = votes.most_common(1)[0]
        avg = float(np.mean(scores_by_name[best_name]))
        return best_name, best_count, avg

    # ============================================================
    # 画中文标签
    # ============================================================

    def _draw_labels(self, img, labels):
        if not labels:
            return img
        img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for text, pos, color_bgr in labels:
            rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text(pos, text, font=self.cn_font, fill=rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ============================================================
    # 发布识别结果
    # ============================================================

    def _publish_results(self, header):
        arr = FaceResultArray()
        arr.header = header

        for info in self.current_frame_info:
            fr = FaceResult()
            fr.name       = str(info["name"])
            fr.score      = float(info["score"])
            fr.raw_score  = float(info["raw_score"])
            fr.bbox_x1    = int(info["bbox"][0])
            fr.bbox_y1    = int(info["bbox"][1])
            fr.bbox_x2    = int(info["bbox"][2])
            fr.bbox_y2    = int(info["bbox"][3])
            fr.confirmed  = bool(info["voted_ok"])
            fr.vote_count = int(info["vote_count"])
            fr.quality    = str(info["quality"])
            arr.faces.append(fr)

        self.result_pub.publish(arr)

    # ============================================================
    # 周期性打印 + 日志
    # ============================================================

    def _periodic_log(self):
        if not self.current_frame_info:
            return
        now = time.time()
        if now - self.last_print_time < self.print_interval:
            return
        self.last_print_time = now

        rospy.loginfo("--- %s 检测到 %d 张人脸 ---"
                      % (time.strftime("%H:%M:%S"), len(self.current_frame_info)))

        for i, info in enumerate(self.current_frame_info):
            emb = info["embedding"]
            rospy.loginfo(
                "  [人脸%d] %s  score=%.4f  vote=%d/%d  confirmed=%s  quality=%s"
                % (i + 1, info["name"], info["score"],
                   info["vote_count"], self.vote_window,
                   info["voted_ok"], info["quality"])
            )
            rospy.loginfo("    query 前8维: %s" % feature_to_str(emb, dim=8))
            if info["db_embedding"] is not None:
                rospy.loginfo("    db    前8维: %s"
                              % feature_to_str(info["db_embedding"], dim=8))
            if self.print_full_feat:
                rospy.loginfo("    query 完整512维: %s"
                              % feature_to_str(emb, dim=None, decimals=5))

        if self.log_fp is not None:
            self.log_fp.write(
                "[%s] faces=%d\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"),
                   len(self.current_frame_info))
            )
            for i, info in enumerate(self.current_frame_info):
                emb = info["embedding"]
                self.log_fp.write(
                    "  人脸%d: %s score=%.4f vote=%d/%d confirmed=%s quality=%s\n"
                    % (i + 1, info["name"], info["score"],
                       info["vote_count"], self.vote_window,
                       info["voted_ok"], info["quality"])
                )
                self.log_fp.write(
                    "    feat[:8]=%s\n"
                    % feature_to_str(emb, dim=8, decimals=5)
                )
                if self.print_full_feat:
                    self.log_fp.write(
                        "    feat_full=%s\n"
                        % feature_to_str(emb, dim=None, decimals=5)
                    )
            self.log_fp.flush()

    # ============================================================
    # 关闭
    # ============================================================

    def on_shutdown(self):
        if self.log_fp is not None:
            self.log_fp.write(
                "===== 会话结束 %s =====\n"
                % time.strftime("%Y-%m-%d %H:%M:%S")
            )
            self.log_fp.close()
            self.log_fp = None
            rospy.loginfo("特征日志已保存: %s" % self.feature_log_file)
        rospy.loginfo("人脸识别节点已退出")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    try:
        node = FaceRecognitionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal("节点启动失败: %s" % e)
        raise
