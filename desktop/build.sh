#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_NAME="SoulX-Singer"
BUILD_ROOT="${TMPDIR:-/tmp}/soulx-singer-desktop-build"
APP_BUNDLE="$BUILD_ROOT/$APP_NAME.app"

rm -rf "$BUILD_ROOT"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

xcrun swiftc \
    -O \
    -framework AppKit \
    -framework WebKit \
    "$SCRIPT_DIR/Sources/main.swift" \
    -o "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

cp "$SCRIPT_DIR/Info.plist" "$APP_BUNDLE/Contents/Info.plist"
# Original icon is excluded. macOS uses a default icon.
chmod 755 "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

codesign --force --deep --sign - "$APP_BUNDLE"
echo "Built $APP_BUNDLE from $PROJECT_DIR"
