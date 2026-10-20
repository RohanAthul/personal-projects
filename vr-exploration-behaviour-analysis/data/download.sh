#!/usr/bin/env bash
# Download the EyeNavGS Rutgers subset (Apache-2.0) at a pinned commit.
# Ding et al. 2025, https://github.com/symmru/EyeNavGS_Rutgers_Dataset
set -euo pipefail

REPO=https://github.com/symmru/EyeNavGS_Rutgers_Dataset.git
COMMIT=7f264fda0223cbc74e191f115ef9fd3b580d8cf4
DEST="$(cd "$(dirname "$0")" && pwd)/raw/EyeNavGS_Rutgers_Dataset"

if [ -d "$DEST/.git" ]; then
    echo "Already present: $DEST"
else
    git clone --quiet "$REPO" "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "EyeNavGS Rutgers at $(git -C "$DEST" rev-parse --short HEAD): $(ls "$DEST"/dataset/*/*.csv | wc -l) trace files"
