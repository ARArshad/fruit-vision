#!/bin/sh
# FruitVision launcher for macOS and Linux -- just double-click this file.
# (On macOS you may need to right-click it once and choose Open.)

cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then
    exec python3 run.py
fi

if command -v python >/dev/null 2>&1; then
    exec python run.py
fi

echo ""
echo "  Python was not found on this computer."
echo ""
echo "  Install it for free from https://www.python.org/downloads/"
echo "  then double-click this file again."
echo ""
printf "  Press Enter to close..."
read -r _
