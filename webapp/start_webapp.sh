#!/usr/bin/env bash
# 启动人脸识别网页（自动加载 ROS 环境，网页打开时按需自动拉起 roslaunch）
#
# 用法:
#   ./start_webapp.sh                 # 默认: 打开网页自动启动相机驱动, 端口 8501
#   ./start_webapp.sh 8501 0          # 第 2 个参数为 0 -> 关闭自动启动 roslaunch
#   FACE_WEBAPP_LAUNCH_TARGET=face_node ./start_webapp.sh
#
# 默认监听 0.0.0.0，同一局域网内的其他电脑可通过 http://<本机IP>:<端口> 访问。
#
# 可用环境变量见 config.py（FACE_WEBAPP_AUTOLAUNCH / FACE_WEBAPP_LAUNCH_TARGET /
# FACE_WEBAPP_LAUNCH_CMD / FACE_WEBAPP_LAUNCH_LOG_DIR ...）

set -euo pipefail

PORT="${1:-8501}"
AUTOLAUNCH="${2:-1}"

ROS_SETUP="${FACE_WEBAPP_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
WS_SETUP="${FACE_WEBAPP_WS_SETUP:-$HOME/catkin_ws/devel/setup.bash}"

# shellcheck disable=SC1090
source "$ROS_SETUP"
if [ -f "$WS_SETUP" ]; then
  # shellcheck disable=SC1090
  source "$WS_SETUP"
fi

export FACE_WEBAPP_AUTOLAUNCH="$AUTOLAUNCH"

cd "$(dirname "$(readlink -f "$0")")"

# 查找局域网 IPv4，仅用于在终端显示访问地址。
LAN_IP="$(hostname -I 2>/dev/null \
  | tr ' ' '\n' \
  | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ && $0 !~ /^127\./ {print; exit}' \
  || true)"
if [ -z "$LAN_IP" ]; then
  LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null \
    | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}' \
    || true)"
fi

echo "本机访问:   http://localhost:${PORT}"
if [ -n "$LAN_IP" ]; then
  echo "其他电脑:   http://${LAN_IP}:${PORT}"
else
  echo "其他电脑:   http://<本机局域网IP>:${PORT}（用 hostname -I 查看本机 IP）"
fi
echo "监听地址:   0.0.0.0:${PORT}   自动启动 roslaunch: ${AUTOLAUNCH}"
exec streamlit run app.py --server.address 0.0.0.0 --server.port "$PORT"
