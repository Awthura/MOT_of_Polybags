#!/bin/bash
# One-time setup so the multi-camera recorder can be run and debugged from an
# agent session without per-command password prompts.
#
# Two independent things block that today, and BOTH need fixing:
#
#   1. sudo needs a password, and an agent shell has no TTY to prompt on.
#      -> fixed by a NOPASSWD sudoers rule, scoped to the two recorder scripts.
#
#   2. Claude Code's permission classifier blocks `sudo ...` Bash calls even
#      when sudo itself would succeed.
#      -> fixed by allow-rules in .claude/settings.local.json.
#
# SECURITY NOTE — read before running.
# This grants passwordless root for two specific scripts. It is scoped to those
# script paths rather than to the interpreter, so it does NOT allow arbitrary
# python as root (a bare `NOPASSWD: .../python` rule would). Python's own flags
# must precede the script path, so extra arguments land in the script's argparse
# rather than the interpreter.
#
# The residual risk, stated plainly: those scripts live in a user-writable repo.
# Anything able to modify them gains silent root. That is inherent to any
# passwordless-sudo arrangement for a script you edit, and is the trade-off you
# accepted. Mitigation if you want it later: `sudo chown root` the two scripts
# so only root can change them.
#
# To revoke everything:
#   sudo rm /etc/sudoers.d/ams_recorder
#   then delete the "Bash(sudo ...)" lines from .claude/settings.local.json
#
# Usage:  bash setup_lab_access.sh

set -euo pipefail

REPO="/Users/awthura/OVGU/AMS"
UTILS="$REPO/real_polybags/real_data/utils"
PY="/opt/anaconda3/envs/ams/bin/python"
SETTINGS="$REPO/.claude/settings.local.json"
SUDOERS_FILE="/etc/sudoers.d/ams_recorder"

echo "=== 1/3  Checking prerequisites ==="
[ -x "$PY" ] || { echo "ERROR: $PY not found or not executable"; exit 1; }
for s in record_all_5_cameras_macos.py reset_realsense.py; do
  [ -f "$UTILS/$s" ] || { echo "ERROR: missing $UTILS/$s"; exit 1; }
done
[ -f "$SETTINGS" ] || { echo "ERROR: missing $SETTINGS"; exit 1; }
echo "  OK"

echo
echo "=== 2/3  Adding Claude Code allow-rules ==="
python3 - "$SETTINGS" "$PY" "$UTILS" <<'PYEOF'
import json, shutil, sys
settings, py, utils = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.copy(settings, settings + ".bak")
with open(settings) as f:
    d = json.load(f)
rules = []
for script in ("record_all_5_cameras_macos.py", "reset_realsense.py"):
    # Both the relative form (run from raw_recordings/) and the absolute form.
    rules.append(f"Bash(sudo {py} ../utils/{script} *)")
    rules.append(f"Bash(sudo {py} {utils}/{script} *)")
rules += [
    "Bash(sudo pkill -9 -f record_all_5_cameras_macos.py)",
    "Bash(sudo pkill -9 -f reset_realsense.py)",
    "Bash(sudo chown awthura *)",
]
allow = d.setdefault("permissions", {}).setdefault("allow", [])
added = [r for r in rules if r not in allow]
allow.extend(added)
with open(settings, "w") as f:
    json.dump(d, f, indent=2)
print(f"  backup: {settings}.bak")
print(f"  added {len(added)} rule(s):" if added else "  all rules already present")
for r in added:
    print("    " + r)
PYEOF

echo
echo "=== 3/3  Installing sudoers rule (asks for your password once) ==="
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<EOF
# Passwordless root for the AMS multi-camera recorder scripts only.
# Installed by real_polybags/real_data/utils/setup_lab_access.sh
# Remove with: sudo rm $SUDOERS_FILE
Cmnd_Alias AMS_RECORD = \\
    $PY $UTILS/record_all_5_cameras_macos.py *, \\
    $PY $UTILS/reset_realsense.py *, \\
    /usr/bin/pkill -9 -f record_all_5_cameras_macos.py, \\
    /usr/bin/pkill -9 -f reset_realsense.py
$(id -un) ALL=(root) NOPASSWD: AMS_RECORD
EOF

# Validate BEFORE installing. A malformed file in /etc/sudoers.d can break sudo
# entirely, which would be a genuinely bad afternoon — visudo -c prevents that.
if ! sudo visudo -c -f "$TMP"; then
  echo "ERROR: generated sudoers file failed validation — NOT installing."
  exit 1
fi

sudo install -m 0440 -o root -g wheel "$TMP" "$SUDOERS_FILE"
echo "  installed $SUDOERS_FILE"

echo
echo "=== Verifying ==="
if sudo -n "$PY" "$UTILS/reset_realsense.py" --help >/dev/null 2>&1; then
  echo "  PASS: passwordless sudo works for the recorder scripts"
else
  echo "  WARN: passwordless sudo did not take effect. Check with:"
  echo "        sudo -l | grep -A3 AMS_RECORD"
fi
if sudo -n /usr/bin/true 2>/dev/null; then
  echo "  NOTE: general passwordless sudo appears active — wider than intended."
  echo "        Check for other rules: sudo -l"
else
  echo "  PASS: general sudo still requires a password (correctly scoped)"
fi

echo
echo "Done. Restart Claude Code so it re-reads settings.local.json."
