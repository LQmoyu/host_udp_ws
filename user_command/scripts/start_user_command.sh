#!/usr/bin/env bash
set -euo pipefail

# Runtime configuration (can be overridden by environment variables).
WS_DIR="${WS_DIR:-/root/udp_ws}"
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-}"
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY_VALUE:-0}"
NODE_NAMESPACE="${NODE_NAMESPACE:-/}"
PARAM_FILE="${PARAM_FILE:-}"

source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
source "${WS_DIR}/install/setup.bash"

if [[ -n "${ROS_DOMAIN_ID_VALUE}" ]]; then
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID_VALUE}"
fi
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY_VALUE}"

if [[ -z "${PARAM_FILE}" ]]; then
  PKG_PREFIX="$(ros2 pkg prefix user_command)"
  INSTALLED_PARAM_FILE="${PKG_PREFIX}/share/user_command/config/param.yaml"
  SRC_PARAM_FILE="${WS_DIR}/src/user_command/config/param.yaml"
  if [[ -f "${INSTALLED_PARAM_FILE}" ]]; then
    PARAM_FILE="${INSTALLED_PARAM_FILE}"
  else
    PARAM_FILE="${SRC_PARAM_FILE}"
  fi
fi

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "[start_user_command] param file not found: ${PARAM_FILE}" >&2
  exit 1
fi

if [[ "${NODE_NAMESPACE}" == "/" ]]; then
  exec ros2 run user_command user_command_node --ros-args --params-file "${PARAM_FILE}"
else
  exec ros2 run user_command user_command_node --ros-args --params-file "${PARAM_FILE}" -r "__ns:=${NODE_NAMESPACE}"
fi

