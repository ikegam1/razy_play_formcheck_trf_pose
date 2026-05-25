#!/bin/bash
set -u

BUCKET="molkky-score-hls"
PREFIX="live"
DIR="./hls_output"
BASE_URL="https://pub-9d9710ab45014df8b3db7b6fa45b8307.r2.dev"

# ここが「ライブより何秒遅らせて公開するか」です。
# v8で4秒セグメントなら 12〜16 秒程度から試すのが良いです。
PUBLISH_DELAY_SEC=12

# シークバーで戻れる時間です。
# 例: 180なら約3分ぶんをm3u8に載せます。
DVR_WINDOW_SEC=180

# v8で hls_time=4 にしている想定。
# もし映像側が2秒セグメントなら 2 にしてください。
SEGMENT_DURATION_SEC=4

STABLE_WAIT_SEC=0.5
PUBLISH_INTERVAL_SEC=2

UPLOADED_LOG="./.uploaded_hls_segments.txt"
touch "$UPLOADED_LOG"

echo "Uploading HLS as DVR-style playlist..."
echo "R2 playlist: ${BASE_URL}/${PREFIX}/stream.m3u8"
echo "PUBLISH_DELAY_SEC=${PUBLISH_DELAY_SEC}"
echo "DVR_WINDOW_SEC=${DVR_WINDOW_SEC}"
echo "SEGMENT_DURATION_SEC=${SEGMENT_DURATION_SEC}"
echo

is_file_stable() {
  local file="$1"
  [ -f "$file" ] || return 1

  local size1 size2
  size1=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
  sleep "$STABLE_WAIT_SEC"
  size2=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)

  [ "$size1" = "$size2" ] && [ "$size1" -gt 0 ]
}

upload_ts_once() {
  local file="$1"
  local name
  name=$(basename "$file")

  [ -f "$file" ] || return 1

  if grep -qx "$name" "$UPLOADED_LOG"; then
    return 0
  fi

  if ! is_file_stable "$file"; then
    return 1
  fi

  wrangler r2 object put "$BUCKET/$PREFIX/$name" \
    --file "$file" \
    --content-type video/mp2t \
    --cache-control "public, max-age=300" \
    --remote >/dev/null

  if [ $? -eq 0 ]; then
    echo "$name" >> "$UPLOADED_LOG"
    echo "[TS] uploaded $name"
    return 0
  fi

  echo "[ERROR] failed to upload $name"
  return 1
}

make_dvr_playlist() {
  local out="$DIR/stream_dvr_publish.m3u8"
  local now
  now=$(date +%s)

  python3 - "$DIR" "$UPLOADED_LOG" "$out" "$PUBLISH_DELAY_SEC" "$DVR_WINDOW_SEC" "$SEGMENT_DURATION_SEC" <<'PY'
import re
import sys
import time
from pathlib import Path

hls_dir = Path(sys.argv[1])
uploaded_log = Path(sys.argv[2])
out_path = Path(sys.argv[3])
delay_sec = float(sys.argv[4])
window_sec = float(sys.argv[5])
seg_duration = float(sys.argv[6])

uploaded = set()
if uploaded_log.exists():
    uploaded = {line.strip() for line in uploaded_log.read_text().splitlines() if line.strip()}

items = []
pattern = re.compile(r"stream_(\d+)\.ts$")

now = time.time()
for p in hls_dir.glob("stream_*.ts"):
    m = pattern.match(p.name)
    if not m:
        continue
    if p.name not in uploaded:
        continue
    try:
        st = p.stat()
    except FileNotFoundError:
        continue
    if st.st_size <= 0:
        continue

    # 直近すぎるTSは公開m3u8に載せない
    age = now - st.st_mtime
    if age < delay_sec:
        continue

    idx = int(m.group(1))
    items.append((idx, p.name, st.st_mtime))

items.sort()

if not items:
    sys.exit(2)

# DVR_WINDOW_SECぶんだけ残す
max_segments = max(3, int(window_sec / seg_duration))
items = items[-max_segments:]

# 少なくとも3本ないと再生開始が不安定
if len(items) < 3:
    sys.exit(2)

first_seq = items[0][0]
target_duration = int(seg_duration + 0.999)

lines = [
    "#EXTM3U",
    "#EXT-X-VERSION:3",
    f"#EXT-X-TARGETDURATION:{target_duration}",
    f"#EXT-X-MEDIA-SEQUENCE:{first_seq}",
    "#EXT-X-PLAYLIST-TYPE:EVENT",
]

for _idx, name, _mtime in items:
    lines.append(f"#EXTINF:{seg_duration:.6f},")
    lines.append(name)

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{items[0][1]} ... {items[-1][1]} ({len(items)} segments)")
PY
}

publish_playlist() {
  local playlist="$DIR/stream_dvr_publish.m3u8"
  [ -f "$playlist" ] || return 1

  wrangler r2 object put "$BUCKET/$PREFIX/stream.m3u8" \
    --file "$playlist" \
    --content-type application/vnd.apple.mpegurl \
    --cache-control "no-cache, no-store, must-revalidate" \
    --remote >/dev/null

  if [ $? -eq 0 ]; then
    echo "[M3U8] published $(grep -cE '\.ts($|\?)' "$playlist") segments"
    return 0
  fi

  return 1
}

last_publish=0

while true; do
  # 1. 新しいTSだけアップロード
  for file in "$DIR"/*.ts; do
    [ -e "$file" ] || continue
    upload_ts_once "$file" || true
  done

  # 2. DVR用の長めのm3u8を作って公開
  now=$(date +%s)
  if [ $((now - last_publish)) -ge "$PUBLISH_INTERVAL_SEC" ]; then
    if range=$(make_dvr_playlist 2>/dev/null); then
      echo "[DVR] $range"
      publish_playlist || true
    else
      echo "[WAIT] not enough safe uploaded segments for DVR playlist"
    fi
    last_publish="$now"
  fi

  sleep 0.5
done
