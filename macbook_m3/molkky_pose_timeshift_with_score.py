import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import urllib.request
import os
import json
import subprocess
import shutil
import threading
import atexit
import signal
from json import JSONDecodeError

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    Image = ImageDraw = ImageFont = None
    PIL_AVAILABLE = False

# --- 解像度設定 ---
WIDTH = 1280
HEIGHT = 720


# --- スコア表示設定 ---
# 別アプリがこのファイルを更新する想定です。
# この録画アプリと同じディレクトリに scores.json を置いてください。
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCORE_FILE = os.path.join(APP_DIR, "scores.json")
SCORE_RELOAD_INTERVAL = 0.5  # 秒。別アプリ側の更新を反映する頻度

# --- HLS配信ファイル出力設定 ---
# ffmpeg がインストールされている場合、ローカルにHLS用 .m3u8 / .ts を生成します。
# 生成先: ./hls_output/stream.m3u8
HLS_ENABLED = True
HLS_DIR = os.path.join(APP_DIR, "hls_output")
HLS_PLAYLIST = os.path.join(HLS_DIR, "stream.m3u8")
UPLOADED_HLS_SEGMENTS_FILE = os.path.join(APP_DIR, ".uploaded_hls_segments.txt")
# HLSはCPU負荷を抑えるため15fpsで出します。
# 30fps指定のまま実処理FPSが落ちると、HLSの時間軸が詰まり、倍速のように見えます。
HLS_FPS = 15
HLS_SEGMENT_SECONDS = 4
HLS_LIST_SIZE = 10

# --- 音声設定 ---
# macOSではffmpegのavfoundation入力を使ってマイク音声を取り込みます。
# うまく音声が入らない場合は、以下でデバイス一覧を確認してください。
#   ffmpeg -f avfoundation -list_devices true -i ""
# 例: 環境変数で指定
#   MOLKKY_AUDIO_INPUT=":0" python3 molkky_pose_timeshift_with_score_v11.py
# 音声は環境変数で無効化できます。
#   MOLKKY_AUDIO_ENABLED=0 python3 molkky_pose_timeshift_with_score_v12.py
AUDIO_ENABLED = os.environ.get("MOLKKY_AUDIO_ENABLED", "1").lower() not in ("0", "false", "no", "off")
AUDIO_INPUT = os.environ.get("MOLKKY_AUDIO_INPUT", ":0")
AUDIO_CODEC = "aac"

# HLS配信では音声トラックの負荷とタイムスタンプずれが固まりの原因になりやすいため、
# まずは低レート・モノラルで安定性を優先します。
AUDIO_BITRATE = os.environ.get("MOLKKY_AUDIO_BITRATE", "32k")
AUDIO_SAMPLE_RATE = os.environ.get("MOLKKY_AUDIO_SAMPLE_RATE", "8000")
AUDIO_CHANNELS = os.environ.get("MOLKKY_AUDIO_CHANNELS", "1")
AUDIO_FILTER = f"aresample=async=1:first_pts=0"

hls_process = None
hls_warned = False

# HLSはメインループから直接書かず、専用スレッドで一定FPS出力する。
# メインループが遅くなっても、最新フレームを複製して実時間の動画にする。
hls_thread = None
hls_stop_event = None
hls_latest_frame = None
hls_frame_lock = threading.Lock()

score_data_cache = {
    "players": []
}
last_score_load_time = 0.0
last_score_mtime = None
last_score_warn_time = 0.0


def find_japanese_font():
    """macOSを中心に、日本語表示できるフォントを探す。"""
    candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
        "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
        "/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Supplemental/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


JP_FONT_PATH = find_japanese_font()
_FONT_CACHE = {}


def get_font(size, bold=False):
    if not PIL_AVAILABLE:
        return None
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        font = ImageFont.truetype(JP_FONT_PATH, size=size) if JP_FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def bgr_to_rgb(color):
    b, g, r = color
    return (r, g, b)


def draw_text(frame, text, org, font_size=20, color=(255, 255, 255), bold=False):
    """OpenCV画像に日本語を描画する。Pillowが無い場合はcv2.putTextへフォールバック。"""
    text = str(text)
    x, y = org
    if PIL_AVAILABLE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        font = get_font(font_size, bold=bold)
        fill = bgr_to_rgb(color)
        draw.text((x, y), text, font=font, fill=fill)
        if bold:
            draw.text((x + 1, y), text, font=font, fill=fill)
        frame[:, :, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    else:
        safe = text.encode("ascii", "replace").decode("ascii")
        scale = max(0.35, font_size / 30.0)
        cv2.putText(frame, safe, (x, y + font_size), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def text_size(text, font_size=20, bold=False):
    """日本語テキストのおおよその描画サイズを返す。"""
    text = str(text)
    if PIL_AVAILABLE:
        font = get_font(font_size, bold=bold)
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    safe = text.encode("ascii", "replace").decode("ascii")
    (w, h), _ = cv2.getTextSize(safe, cv2.FONT_HERSHEY_SIMPLEX, max(0.35, font_size / 30.0), 1)
    return w, h


def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --- AIモデルファイルの設定 ---
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    print(f"[INFO] 骨格推定用のAIモデル({MODEL_PATH})をダウンロード中...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- 最新のMediaPipe Tasks APIの初期化 ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)

# 骨格の接続定義
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
]

# --- カメラ設定 ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# --- 状態管理変数 ---
frame_buffer = deque()            # 過去映像ストック用バッファ（最大120秒）
is_playback = False               # 遅延再生中フラグ
playback_remaining_time = 0.0     # 【修正】遅延の残り秒数（カウントダウン用）
playback_anchor_time = 0.0        # 【新設】遅延開始時の基準システム時刻

is_recording = False              # 録画中フラグ
video_writer = None               # 動画書き込み用オブジェクト（ffmpegが使えない場合のフォールバック）
record_process = None             # 音声付き録画用ffmpegプロセス
record_uses_ffmpeg = False
rec_start_time = 0.0              # 録画開始時間

# 実測FPS計測用
fps_timestamps = deque(maxlen=30)
current_fps = 30.0

# UIボタン配置 (x1, x2, アクション種別, 設定値, ラベル)
BUTTONS = [
    (10, 90,   "sub",   2,  "< 2s"),
    (100, 180, "sub",   5,  "< 5s"),
    (190, 270, "sub",   10, "< 10s"),
    (280, 360, "sub",   30, "< 30s"),
    (370, 450, "add",   5,  "5s >"),
    (460, 540, "add",   1,  "1s >"),
    (550, 630, "reset", None, "LIVE"), # ↩ボタンの代用（通常再生リセット）
    (660, 740, "rec_start", None, "REC"),
    (750, 830, "rec_stop", None, "STOP")
]

BUTTON_Y1 = HEIGHT - 58
BUTTON_Y2 = HEIGHT - 18

# マウスクリックイベントの処理
def on_mouse_click(event, x, y, flags, param):
    global is_playback, frame_buffer, playback_remaining_time, playback_anchor_time
    global is_recording, video_writer, record_process, record_uses_ffmpeg, rec_start_time
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if BUTTON_Y1 <= y <= BUTTON_Y2:
            current_time = time.time()
            for x1, x2, btn_type, val, label in BUTTONS:
                if x1 <= x <= x2:
                    # 1. 過去に巻き戻すボタン（遅延残り時間を加算）
                    if btn_type == "sub":
                        if not is_playback:
                            is_playback = True
                            playback_anchor_time = current_time
                            playback_remaining_time = float(val)
                        else:
                            playback_remaining_time += float(val)
                        
                        if playback_remaining_time > 120.0:
                            playback_remaining_time = 120.0  # 上限120秒ガード
                        print(f"[INFO] 遅延残り時間を変更: {playback_remaining_time:.1f}秒")
                    
                    # 2. 未来に進めるボタン（遅延残り時間を減算）
                    elif btn_type == "add":
                        if is_playback:
                            playback_remaining_time -= float(val)
                            if playback_remaining_time <= 0:
                                is_playback = False
                                playback_remaining_time = 0.0
                                print("[INFO] 通常再生（ライブ）に戻りました。")
                            else:
                                print(f"[INFO] 遅延残り時間を変更: {playback_remaining_time:.1f}秒")
                    
                    # 3. LIVE（↩）ボタン：通常再生にリセット
                    elif btn_type == "reset":
                        is_playback = False
                        playback_remaining_time = 0.0
                        print("[INFO] 通常再生（ライブ）に戻りました。")
                    
                    # 4. 録画開始ボタン
                    elif btn_type == "rec_start":
                        if not is_recording:
                            is_recording = True
                            rec_start_time = current_time
                            filename = f"pose_rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                            
                            video_fps = max(10.0, min(60.0, current_fps))
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(filename, fourcc, video_fps, (WIDTH, HEIGHT))
                            print(f"\n[INFO] 録画を開始しました (設定FPS: {video_fps:.1f}): {filename}")
                    
                    # 5. 録画停止ボタン
                    elif btn_type == "rec_stop":
                        if is_recording:
                            is_recording = False
                            if video_writer is not None:
                                video_writer.release()
                                video_writer = None
                            print("\n[INFO] 録画を停止・動画ファイルを保存しました。")
                    break


# --- スコア読み込み・描画 ---
def warn_score_load_error(message, current_time):
    """scores.json 関連の警告を出しすぎないようにする。"""
    global last_score_warn_time
    if current_time - last_score_warn_time >= 5.0:
        print(f"[WARN] scores.json の読み込みに失敗しました: {message}")
        last_score_warn_time = current_time


def load_scores_if_needed(current_time):
    """scores.json を定期的に読み込み、最大4名分のスコア情報をキャッシュする。

    v4スコアアプリの拡張項目にも対応:
    - score が "OUT" の場合はそのまま表示
    - score_value / set_count / miss_count / is_out を読み込む
    - meta.current_player / set_number / order_mode も保持する
    """
    global score_data_cache, last_score_load_time, last_score_mtime

    if current_time - last_score_load_time < SCORE_RELOAD_INTERVAL:
        return score_data_cache

    last_score_load_time = current_time

    if not os.path.exists(SCORE_FILE):
        return score_data_cache

    try:
        mtime = os.path.getmtime(SCORE_FILE)
        if last_score_mtime == mtime:
            return score_data_cache

        with open(SCORE_FILE, "r", encoding="utf-8-sig") as f:
            raw_text = f.read()

        if not raw_text.strip():
            warn_score_load_error("scores.json が空です。別アプリの書き込み途中かもしれません。", current_time)
            return score_data_cache

        loaded = json.loads(raw_text)

        players = loaded.get("players", [])
        if not isinstance(players, list):
            players = []

        normalized_players = []
        for p in players[:4]:
            if not isinstance(p, dict):
                continue

            name = str(p.get("name", ""))[:16]
            is_out = bool(p.get("is_out", False))

            raw_score = p.get("score", 0)
            if is_out or str(raw_score).upper() == "OUT":
                score_display = "OUT"
                score_value = to_int(p.get("score_value", 0), 0)
                is_out = True
            else:
                score_value = to_int(p.get("score_value", raw_score), 0)
                score_display = str(score_value)

            recent_scores = p.get("recent_scores", [])
            if not isinstance(recent_scores, list):
                recent_scores = []
            recent_scores = recent_scores[:3]
            recent_scores = ["" if s is None else str(s) for s in recent_scores]
            while len(recent_scores) < 3:
                recent_scores.append("")

            miss_count = max(0, min(3, to_int(p.get("miss_count", 0), 0)))
            set_count = max(0, to_int(p.get("set_count", 0), 0))

            normalized_players.append({
                "name": name,
                "score": score_display,
                "score_value": score_value,
                "recent_scores": recent_scores,
                "set_count": set_count,
                "miss_count": miss_count,
                "is_out": is_out,
            })

        meta = loaded.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}

        score_data_cache = {
            "players": normalized_players,
            "meta": {
                "current_player": str(meta.get("current_player", "")),
                "turn_count": to_int(meta.get("turn_count", 1), 1),
                "set_number": to_int(meta.get("set_number", 1), 1),
                "order_mode": str(meta.get("order_mode", "")),
            }
        }
        last_score_mtime = mtime

    except JSONDecodeError as e:
        line = ""
        try:
            lines = raw_text.splitlines()
            if 1 <= e.lineno <= len(lines):
                line = f" / 該当行: {lines[e.lineno - 1]}"
        except Exception:
            pass
        warn_score_load_error(f"JSON形式が不正です。line {e.lineno}, column {e.colno}: {e.msg}{line}", current_time)
    except Exception as e:
        warn_score_load_error(str(e), current_time)

    return score_data_cache

def draw_score_overlay(frame, scores):
    """画面右上ぎりぎりに、コンパクトなスコア表を描画する。日本語名、OUT、ミスX、セットカウントに対応。"""
    players = scores.get("players", []) if isinstance(scores, dict) else []
    meta = scores.get("meta", {}) if isinstance(scores, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    if not players:
        return frame

    visible_players = players[:4]
    current_player_name = str(meta.get("current_player", ""))
    set_number = meta.get("set_number", 1)

    # タイトルを無くした分、全体をコンパクト化する。
    row_h = 36
    padding_x = 12
    padding_y = 8
    header_h = 42
    table_w = 560
    table_h = header_h + row_h * len(visible_players) + padding_y

    x1 = WIDTH - 20
    x0 = x1 - table_w
    y0 = 4  # 画面上部ぎりぎり
    y1 = y0 + table_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)

    # Reverse / Slide は表示しない。SET表記は小さくする。
    draw_text(frame, f"SET {set_number}", (x0 + padding_x, y0 + 6), 11, (230, 230, 230), bold=True)

    col_name = x0 + 14
    col_set = x0 + 230
    col_score = x0 + 290
    box_w = 48
    box_h = 26
    box_gap = 12
    box_x0 = x0 + 390
    recent_box_x = [box_x0 + i * (box_w + box_gap) for i in range(3)]

    header_y = y0 + 24
    draw_text(frame, "PLAYER", (col_name, header_y), 15, (220, 220, 220), bold=True)
    draw_text(frame, "SET", (col_set, header_y), 15, (220, 220, 220), bold=True)
    draw_text(frame, "TOTAL", (col_score, header_y), 15, (220, 220, 220), bold=True)
    draw_text(frame, "LAST 3", (box_x0, header_y), 15, (220, 220, 220), bold=True)
    line_y = header_y + 20
    cv2.line(frame, (x0 + 10, line_y), (x1 - 10, line_y), (180, 180, 180), 1)

    start_y = line_y + 23
    for i, player in enumerate(visible_players):
        row_center_y = start_y + i * row_h
        name = player.get("name", "")
        score_display = player.get("score", player.get("score_value", 0))
        recent_scores = player.get("recent_scores", ["", "", ""])[:3]
        while len(recent_scores) < 3:
            recent_scores.append("")

        set_count = player.get("set_count", 0)
        miss_count = max(0, min(3, to_int(player.get("miss_count", 0), 0)))
        is_out = bool(player.get("is_out", False)) or str(score_display).upper() == "OUT"
        is_current = current_player_name and str(name) == current_player_name

        name_color = (0, 255, 255) if is_current and not is_out else (255, 255, 255)
        if is_out:
            name_color = (170, 170, 170)

        prefix = ">" if is_current and not is_out else " "
        display_name = prefix + str(name)[:12]
        draw_text(frame, display_name, (col_name, row_center_y - 13), 18, name_color, bold=True)

        name_w, _ = text_size(display_name, 18, bold=True)
        miss_x = min(col_set - 42, col_name + name_w + 8)
        for m in range(miss_count):
            draw_text(frame, "X", (miss_x + m * 14, row_center_y - 11), 16, (40, 40, 230), bold=True)

        draw_text(frame, str(set_count), (col_set + 4, row_center_y - 13), 18, (255, 255, 255), bold=True)
        if is_out:
            draw_text(frame, "OUT", (col_score - 4, row_center_y - 13), 18, (170, 170, 170), bold=True)
        else:
            draw_text(frame, str(score_display), (col_score, row_center_y - 15), 22, (0, 255, 255), bold=True)

        for j, s_value in enumerate(recent_scores):
            cell_x = recent_box_x[j]
            cell_y1 = row_center_y - box_h // 2
            cell_y2 = row_center_y + box_h // 2
            cv2.rectangle(frame, (cell_x, cell_y1), (cell_x + box_w, cell_y2), (70, 70, 70), -1)
            cv2.rectangle(frame, (cell_x, cell_y1), (cell_x + box_w, cell_y2), (170, 170, 170), 1)
            text = str(s_value)
            text_x = cell_x + 14
            if len(text) >= 2:
                text_x = cell_x + 8
            draw_text(frame, text, (text_x, row_center_y - 10), 18, (255, 255, 255), bold=True)

    return frame


def start_recording_writer(filename, video_fps):
    """録画を開始する。ffmpegが使える場合はマイク音声付きMP4、失敗時は映像のみへフォールバック。"""
    global video_writer, record_process, record_uses_ffmpeg

    stop_recording_writer()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        cmd = [
            ffmpeg_path,
            "-loglevel", "warning",
            "-y",
            "-thread_queue_size", "1024",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(video_fps),
            "-i", "-",
        ]

        if AUDIO_ENABLED:
            cmd += [
                "-thread_queue_size", "1024",
                "-f", "avfoundation",
                "-i", AUDIO_INPUT,
                "-map", "0:v:0",
                "-map", "1:a:0",
            ]
        else:
            cmd += ["-map", "0:v:0"]

        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
        ]

        if AUDIO_ENABLED:
            cmd += [
                "-af", AUDIO_FILTER,
                "-c:a", AUDIO_CODEC,
                "-b:a", AUDIO_BITRATE,
                "-ar", AUDIO_SAMPLE_RATE,
                "-ac", AUDIO_CHANNELS,
                "-shortest",
            ]
        else:
            cmd += ["-an"]

        cmd += ["-movflags", "+faststart", filename]

        try:
            record_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            record_uses_ffmpeg = True
            print(f"[INFO] 音声付き録画を開始しました。音声入力: {AUDIO_INPUT if AUDIO_ENABLED else 'なし'}")
            return
        except Exception as e:
            print(f"[WARN] ffmpeg録画を開始できませんでした。映像のみで録画します: {e}")
            record_process = None
            record_uses_ffmpeg = False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, video_fps, (WIDTH, HEIGHT))
    record_uses_ffmpeg = False
    print("[WARN] 映像のみの録画です。音声を入れるには ffmpeg とマイク入力設定が必要です。")


def write_recording_frame(frame):
    """録画中のフレームを書き込む。"""
    global record_process, video_writer, is_recording

    if record_uses_ffmpeg and record_process is not None and record_process.stdin is not None:
        try:
            record_process.stdin.write(frame.tobytes())
            return
        except (BrokenPipeError, OSError) as e:
            print(f"[WARN] 音声付き録画プロセスが停止しました: {e}")
            try:
                record_process.stdin.close()
            except Exception:
                pass
            record_process = None
            is_recording = False
            return

    if video_writer is not None:
        video_writer.write(frame)


def stop_recording_writer():
    """録画を停止する。"""
    global video_writer, record_process, record_uses_ffmpeg

    if record_process is not None:
        try:
            if record_process.stdin:
                record_process.stdin.close()
            record_process.wait(timeout=3)
        except Exception:
            try:
                record_process.terminate()
                record_process.wait(timeout=1)
            except Exception:
                pass
        record_process = None

    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass
        video_writer = None

    record_uses_ffmpeg = False

def start_hls_writer():
    """ffmpegでHLS用ファイルをローカル生成する。"""
    global hls_process, hls_warned, hls_thread, hls_stop_event, hls_latest_frame
    hls_latest_frame = None
    if not HLS_ENABLED:
        return None
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        if not hls_warned:
            print("[WARN] ffmpeg が見つからないためHLS出力を無効化します。brew install ffmpeg で有効化できます。")
            hls_warned = True
        return None

    os.makedirs(HLS_DIR, exist_ok=True)
    for name in os.listdir(HLS_DIR):
        if name.endswith((".m3u8", ".ts", ".tmp")):
            try:
                os.remove(os.path.join(HLS_DIR, name))
            except OSError:
                pass

    cmd = [
        ffmpeg_path,
        "-loglevel", "warning",
        "-y",
        "-thread_queue_size", "1024",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(HLS_FPS),
        "-i", "-",
    ]

    if AUDIO_ENABLED:
        # macOSのマイク音声を同時に取り込む。AUDIO_INPUTは通常 ":0"。
        cmd += [
            "-thread_queue_size", "1024",
            "-f", "avfoundation",
            "-i", AUDIO_INPUT,
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]
    else:
        cmd += ["-map", "0:v:0"]

    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(HLS_FPS * HLS_SEGMENT_SECONDS),
        "-sc_threshold", "0",
    ]

    if AUDIO_ENABLED:
        cmd += [
            "-af", AUDIO_FILTER,
            "-c:a", AUDIO_CODEC,
            "-b:a", AUDIO_BITRATE,
            "-ar", AUDIO_SAMPLE_RATE,
            "-ac", AUDIO_CHANNELS,
        ]
    else:
        cmd += ["-an"]

    cmd += [
        "-f", "hls",
        "-hls_time", str(HLS_SEGMENT_SECONDS),
        "-hls_list_size", str(HLS_LIST_SIZE),
        "-hls_flags", "delete_segments+append_list+omit_endlist",
        "-hls_segment_filename", os.path.join(HLS_DIR, "stream_%05d.ts"),
        HLS_PLAYLIST,
    ]
    try:
        hls_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        hls_stop_event = threading.Event()
        hls_thread = threading.Thread(target=hls_frame_publisher, daemon=True)
        hls_thread.start()
        if AUDIO_ENABLED:
            print(f"[INFO] HLS出力を開始しました: {HLS_PLAYLIST} ({HLS_FPS}fps, 音声: 有効 {AUDIO_INPUT}, {AUDIO_SAMPLE_RATE}Hz, {AUDIO_CHANNELS}ch, {AUDIO_BITRATE})")
        else:
            print(f"[INFO] HLS出力を開始しました: {HLS_PLAYLIST} ({HLS_FPS}fps, 音声: 無効)")
    except Exception as e:
        print(f"[WARN] HLS出力を開始できませんでした: {e}")
        hls_process = None
    return hls_process


def write_hls_frame_raw(frame):
    """HLS出力へ1フレームを書き込む。失敗時は自動停止。"""
    global hls_process
    if hls_process is None or hls_process.stdin is None:
        return False
    try:
        hls_process.stdin.write(frame.tobytes())
        return True
    except (BrokenPipeError, OSError) as e:
        print(f"[WARN] HLS出力が停止しました: {e}")
        try:
            hls_process.stdin.close()
        except Exception:
            pass
        hls_process = None
        return False


def hls_frame_publisher():
    """最新フレームをHLS_FPSでffmpegへ送り続ける専用スレッド。"""
    global hls_process

    interval = 1.0 / float(HLS_FPS)
    next_time = time.monotonic()
    last_frame = None

    while hls_stop_event is not None and not hls_stop_event.is_set():
        now = time.monotonic()
        if now < next_time:
            time.sleep(min(0.01, next_time - now))
            continue

        with hls_frame_lock:
            if hls_latest_frame is not None:
                last_frame = hls_latest_frame.copy()

        # まだ最初のフレームが来ていない間は何も書かない。
        if last_frame is not None:
            if not write_hls_frame_raw(last_frame):
                return

        next_time += interval

        # ffmpeg書き込み等で遅れた場合、追いつきのために大量連投せず現在時刻へ寄せる。
        # これによりCPU詰まり時も「倍速で追いかける」動きを避ける。
        if time.monotonic() - next_time > 1.0:
            next_time = time.monotonic() + interval


def update_hls_latest_frame(frame):
    """メインループから最新フレームだけを共有する。"""
    global hls_latest_frame
    if hls_process is None:
        return
    with hls_frame_lock:
        hls_latest_frame = frame.copy()


def write_hls_frame_realtime(frame, now=None):
    """互換用。v9では専用スレッドへ最新フレームを渡すだけ。"""
    update_hls_latest_frame(frame)


def stop_hls_writer():
    """HLS用ffmpegと送信スレッドを停止する。"""
    global hls_process, hls_stop_event, hls_thread

    if hls_stop_event is not None:
        hls_stop_event.set()

    if hls_thread is not None and hls_thread.is_alive():
        try:
            hls_thread.join(timeout=1.5)
        except Exception:
            pass
    hls_thread = None

    if hls_process is not None:
        try:
            if hls_process.stdin:
                hls_process.stdin.close()
            hls_process.wait(timeout=2)
        except Exception:
            try:
                hls_process.terminate()
                hls_process.wait(timeout=1)
            except Exception:
                pass
    hls_process = None


def cleanup_hls_runtime_files():
    """次回起動時に古いアップロード状態やHLS断片が残らないよう掃除する。"""
    try:
        if os.path.exists(UPLOADED_HLS_SEGMENTS_FILE):
            os.remove(UPLOADED_HLS_SEGMENTS_FILE)
            print(f"[INFO] HLSアップロードログを削除しました: {UPLOADED_HLS_SEGMENTS_FILE}")
    except OSError as e:
        print(f"[WARN] HLSアップロードログを削除できませんでした: {e}")

    try:
        os.makedirs(HLS_DIR, exist_ok=True)
        for name in os.listdir(HLS_DIR):
            path = os.path.join(HLS_DIR, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except OSError as e:
                print(f"[WARN] HLS出力ファイルを削除できませんでした: {path}: {e}")
        print(f"[INFO] HLS出力ディレクトリを空にしました: {HLS_DIR}")
    except OSError as e:
        print(f"[WARN] HLS出力ディレクトリを掃除できませんでした: {e}")


def shutdown_cleanup():
    """アプリ終了時の共通クリーンアップ。"""
    stop_recording_writer()
    stop_hls_writer()
    cleanup_hls_runtime_files()

WINDOW_NAME = 'mac_pose_timeshift'
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

print("Starting live inference. Press 'q' to quit, 'ESC' or 'BackSpace' to return to Live.")
print(f"[INFO] スコア表示ファイル: {SCORE_FILE}")
print(f"[INFO] 音声入力: {'有効 ' + AUDIO_INPUT if AUDIO_ENABLED else '無効'}")
if not PIL_AVAILABLE:
    print("[WARN] Pillowが無いため日本語表示は?になります: python3 -m pip install pillow")
elif JP_FONT_PATH is None:
    print("[WARN] 日本語フォントが見つかりません。macOS標準フォントを確認してください。")
atexit.register(shutdown_cleanup)

def _handle_exit_signal(signum, frame):
    print(f"[INFO] 終了シグナルを受信しました: {signum}")
    shutdown_cleanup()
    raise SystemExit(0)

for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _handle_exit_signal)
    except Exception:
        pass

start_hls_writer()
last_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できません。")
        break
        
    current_time = time.time()
    dt = current_time - last_time  # 前回のループからの経過時間を計算
    last_time = current_time
    
    # 実測FPSの計算
    fps_timestamps.append(current_time)
    if len(fps_timestamps) > 1:
        total_time = fps_timestamps[-1] - fps_timestamps[0]
        if total_time > 0:
            current_fps = (len(fps_timestamps) - 1) / total_time

    frame = cv2.flip(frame, 1)  # 鏡像表示
    
    # 骨格推定の実行
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    display_frame = frame.copy()
    
    # 骨格推定の描画
    if results.pose_landmarks:
        for landmark_list in results.pose_landmarks:
            for start_idx, end_idx in CONNECTIONS:
                if start_idx < len(landmark_list) and end_idx < len(landmark_list):
                    lm_start = landmark_list[start_idx]
                    lm_end = landmark_list[end_idx]
                    x1, y1 = int(lm_start.x * WIDTH), int(lm_start.y * HEIGHT)
                    x2, y2 = int(lm_end.x * WIDTH), int(lm_end.y * HEIGHT)
                    if 0 <= x1 < WIDTH and 0 <= y1 < HEIGHT and 0 <= x2 < WIDTH and 0 <= y2 < HEIGHT:
                        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for lm in landmark_list:
                cx, cy = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    cv2.circle(display_frame, (cx, cy), 4, (0, 255, 0), -1)
                    
    # スコア表示を重ねる（録画・遅延再生バッファにも反映）
    scores = load_scores_if_needed(current_time)
    draw_score_overlay(display_frame, scores)

    # 【録画】裏で常に最新映像（骨格・スコア描画済み）を保存
    if is_recording:
        write_recording_frame(display_frame)
                    
    # 【大容量化】過去「120秒間」のライブフレームをバッファに常時ストック
    frame_buffer.append((current_time, display_frame.copy()))
    while frame_buffer and (current_time - frame_buffer[0][0] > 120.0):
        frame_buffer.popleft()
        
    # 【修正】遅延再生中のカウントダウンおよびフレーム検索処理
    if is_playback:
        playback_remaining_time -= dt  # 時間の経過とともに残り秒数を減らす
        
        if playback_remaining_time <= 0:
            is_playback = False
            playback_remaining_time = 0.0
            print("[INFO] 再生が最新に追いついたため、通常再生（ライブ）に戻ります。")
            out_frame = display_frame.copy()
        else:
            # ターゲットとする過去の絶対時刻を割り出す
            target_time = playback_anchor_time - playback_remaining_time
            best_frame = None
            
            # 時系列バッファを最新から過去へ逆引き検索し、目標時刻に最も近いコマを特定
            for t, f in reversed(frame_buffer):
                if t <= target_time:
                    best_frame = f
                    break
            
            if best_frame is None and len(frame_buffer) > 0:
                best_frame = frame_buffer[0][1]
                
            if best_frame is not None:
                out_frame = best_frame.copy()
                # 綺麗にカウントダウンさせるため、残り秒数を切り上げて整数表記
                display_sec = int(np.ceil(playback_remaining_time))
                cv2.putText(out_frame, f"-{display_sec}sec PLAYBACK", (WIDTH - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                out_frame = display_frame.copy()
    else:
        out_frame = display_frame.copy()
        
    # 録画時間表示
    if is_recording:
        elapsed_time = current_time - rec_start_time
        rec_min = int(elapsed_time // 60)
        rec_sec = int(elapsed_time % 60)
        cv2.putText(out_frame, f"REC {rec_min:02d}:{rec_sec:02d}", (WIDTH - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # UIボタンの描画処理
    for x1, x2, btn_type, val, label in BUTTONS:
        if btn_type == "rec_start" and is_recording:
            bg_color = (0, 0, 255)       
            text_color = (255, 255, 255)
        elif btn_type == "reset" and is_playback:
            bg_color = (0, 200, 0)       # 遅延再生中はLIVEボタンを緑に
            text_color = (255, 255, 255)
        elif btn_type == "add" and not is_playback:
            bg_color = (180, 180, 180)   # ライブ中は「未来に進む」ボタンを無効化（グレー）
            text_color = (120, 120, 120)
        else:
            bg_color = (220, 220, 220)   
            text_color = (0, 0, 0)
            
        cv2.rectangle(out_frame, (x1, BUTTON_Y1), (x2, BUTTON_Y2), bg_color, -1)
        cv2.rectangle(out_frame, (x1, BUTTON_Y1), (x2, BUTTON_Y2), (100, 100, 100), 2)
        
        if len(label) == 3:
            offset_x = 22
        elif len(label) == 4:
            offset_x = 16
        else:
            offset_x = 10
        cv2.putText(out_frame, label, (x1 + offset_x, BUTTON_Y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # 動作FPS表示
    cv2.putText(out_frame, f"FPS: {current_fps:.1f}", (WIDTH - 120, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    write_hls_frame_realtime(out_frame, current_time)

    cv2.imshow(WINDOW_NAME, out_frame)
    
    # 【新機能】キーボード入力処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # ESCキー (27) または BackSpace/Deleteキー (8 または Mac用127) で通常（LIVE）に戻る
    elif key in [27, 8, 127]:
        is_playback = False
        playback_remaining_time = 0.0
        print("[INFO] キーボード入力により通常再生（ライブ）に戻りました。")

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
shutdown_cleanup()
