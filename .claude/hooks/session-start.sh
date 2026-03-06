#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

REPO="${CLAUDE_PROJECT_DIR:-$(pwd)}"

echo "[session-start] Installing test dependencies..."
pip install --quiet --disable-pip-version-check \
  -r "$REPO/requirements-test.txt"

echo "[session-start] Setting PYTHONPATH..."
echo "export PYTHONPATH=\"$REPO\"" >> "$CLAUDE_ENV_FILE"

echo "[session-start] Done."
