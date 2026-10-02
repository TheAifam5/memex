#!/usr/bin/env bash
# Check the actual linked SDK, which selects AppKit's compatibility behavior.
# Pass an existing app executable to check it without rebuilding.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -gt 0 ]]; then
  binary=$1
else
  "$ROOT/scripts/build.sh" debug
  binary="$ROOT/build/Memex.app/Contents/MacOS/Memex"
fi
sdk_version=$(xcrun --sdk macosx --show-sdk-version)
otool -l "$binary" | awk -v expected_sdk="$sdk_version" '
  function version(value, parts) {
    split(value, parts, ".")
    return sprintf("%d.%d.%d", parts[1], parts[2], parts[3])
  }
  $1 == "cmd" { in_build_version = $2 == "LC_BUILD_VERSION" }
  in_build_version && $1 == "minos" && version($2) != version("14.0") {
    print "Expected macOS 14.0 deployment target, got " $2 > "/dev/stderr"
    failed = 1
  }
  in_build_version && $1 == "sdk" {
    count++
    if (version($2) != version(expected_sdk)) {
      print "Expected linked SDK " expected_sdk ", got " $2 > "/dev/stderr"
      failed = 1
    }
  }
  END {
    if (!count) {
      print "Missing LC_BUILD_VERSION" > "/dev/stderr"
      failed = 1
    }
    exit failed
  }
'
echo "App records SDK $sdk_version and retains macOS 14.0 support"
