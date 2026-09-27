#!/usr/bin/env bash
# Clone the official CLM repository at the pinned commit into external/CLM.
# Usage: tools/setup_clm.sh [--install]   (--install also runs `pip install -e` on it)
set -euo pipefail
COMMIT="bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CLM_ROOT:-$HERE/external/CLM}"

if [ ! -d "$DEST/.git" ]; then
    git clone --quiet https://github.com/Contrastive-LM/CLM "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$COMMIT"
echo "official CLM at $DEST @ $(git -C "$DEST" rev-parse HEAD)"

if [ "${1:-}" = "--install" ]; then
    pip install -q -e "$DEST"
    pip install -q -r "$DEST/train/requirements.txt"
fi
