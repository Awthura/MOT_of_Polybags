#!/bin/bash
# SUDO_ASKPASS helper — lets sudo be driven from a non-interactive session.
#
# Why: the recording scripts need root (librealsense must claim the RealSense
# USB devices on macOS), but an agent/automation shell has no controlling
# terminal, so plain `sudo` fails with:
#     sudo: a terminal is required to read the password
#
# With this helper, `sudo -A <cmd>` pops a native macOS password dialog. The
# password goes from the dialog straight into sudo — it is never echoed to the
# terminal, written to disk, logged, or visible to the calling process. No
# password is stored in this file.
#
# Usage:
#     export SUDO_ASKPASS=/Users/awthura/OVGU/AMS/real_polybags/real_data/utils/sudo_askpass.sh
#     sudo -A /opt/anaconda3/envs/ams/bin/python ../utils/record_all_5_cameras_macos.py --duration 10
#
# sudo caches credentials for ~5 minutes by default, so one dialog covers a
# burst of commands. To make one prompt last a whole lab session, raise the
# timeout (asks for your password once, in a real terminal):
#     sudo sh -c 'echo "Defaults timestamp_timeout=180" > /etc/sudoers.d/ams_timeout && chmod 440 /etc/sudoers.d/ams_timeout'
#
# Note: macOS sudo uses per-terminal credential caching (tty_tickets), so a
# password entered in Terminal.app does NOT carry over to an agent session.
# That is why this helper exists rather than just priming sudo beforehand.
osascript \
  -e 'display dialog "sudo password (multi-camera recorder):" default answer "" with hidden answer with title "Claude Code — sudo"' \
  -e 'text returned of result' 2>/dev/null
