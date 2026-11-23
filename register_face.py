#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# 照片目录（用你实际的路径）
PHOTO_DIR = "/home/xiangsi/catkin_ws/src/face_recognition_ros/face_data/chenzhen"

# 数据库输出路径
DATABASE_FILE = "/home/xiangsi/catkin_ws/src/face_recognition_ros/model/chenzhen_face_database.pkl"

PERSON_NAME = "chenzhen"

print("照片目录:", PHOTO_DIR)
print("数据库路径:", DATABASE_FILE)

# 检查照片目录
if not os.path.isdir(PHOTO_DIR):
    raise RuntimeError("照片目录不存在: %s" % PHOTO_DIR)

files = [f for f in sorted(os.listdir(PHOTO_DIR))
         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
print("找到 %d 张图片" % len(files))
if not files:
    raise RuntimeError("照片目录里没有图片")

# 加载 InsightFace
print("正在加载 InsightFace...")
app = FaceAnalysis(
    name="buffalo_l",
    root=os.path.expanduser("~/.insightface"),
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace 加载完成\n")

embeddings = []
for fn in files:
    path = os.path.join(PHOTO_DIR, fn)
    img = cv2.imread(path)
    if img is None:
        print("读取失败:", fn)
        continue
    faces = app.get(img)
    if len(faces) == 0:
        print("跳过 %s：未检测到人脸" % fn)
        continue
    if len(faces) > 1:
        print("跳过 %s：检测到 %d 张人脸" % (fn, len(faces)))
        continue
    emb = faces[0].embedding.astype(np.float32)
    emb = emb / max(np.linalg.norm(emb), 1e-6)
    embeddings.append(emb)
    print("已提取: %s  det_score=%.3f" % (fn, faces[0].det_score))

if not embeddings:
    raise RuntimeError("没有提取到任何人脸特征")

# 平均模板
template = np.mean(embeddings, axis=0).astype(np.float32)
template = template / max(np.linalg.norm(template), 1e-6)

db = {
    PERSON_NAME: {
        "embeddings": embeddings,
        "template": template,
    }
}

os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
with open(DATABASE_FILE, "wb") as f:
    pickle.dump(db, f)

print("")
print("=" * 60)
print("数据库已保存:", DATABASE_FILE)
print("人数:", len(db), " 特征数:", len(embeddings))
print("NumPy 版本:", np.__version__)
print("=" * 60)
