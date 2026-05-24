import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import urllib.request
import os
import json
from json import JSONDecodeError

# --- 解像度設定 ---
WIDTH = 1280
HEIGHT = 720


# --- スコア表示設定 ---
# 別アプリがこのファイルを更新する想定です。
# この録画アプリと同じディレクトリに scores.json を置いてください。
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCORE_FILE = os.path.join(APP_DIR, "scores.json")
SCORE_RELOAD_INTERVAL = 0.5  # 秒。別アプリ側の更新を反映する頻度

score_data_cache = {
    "players": []
}
last_score_load_time = 0.0
last_score_mtime = None
last_score_warn_time = 0.0

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
video_writer = None               # 動画書き込み用オブジェクト
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

# マウスクリックイベントの処理
def on_mouse_click(event, x, y, flags, param):
    global is_playback, frame_buffer, playback_remaining_time, playback_anchor_time
    global is_recording, video_writer, rec_start_time
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= y <= 50:
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

    別アプリが scores.json を上書きしている瞬間に読み込むと、
    空ファイルや途中までのJSONを読んでしまうことがあります。
    その場合は前回の正常なスコア表示を維持します。
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

            name = str(p.get("name", ""))[:12]

            try:
                score = int(p.get("score", 0))
            except (TypeError, ValueError):
                score = 0

            recent_scores = p.get("recent_scores", [])
            if not isinstance(recent_scores, list):
                recent_scores = []
            recent_scores = recent_scores[-3:]
            recent_scores = ["" if s is None else str(s) for s in recent_scores]

            while len(recent_scores) < 3:
                recent_scores.insert(0, "")

            normalized_players.append({
                "name": name,
                "score": score,
                "recent_scores": recent_scores
            })

        score_data_cache = {
            "players": normalized_players
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
    """画面左下にスコア表を描画する。"""
    players = scores.get("players", []) if isinstance(scores, dict) else []
    if not players:
        return frame

    visible_players = players[:4]

    # レイアウトを少し大きめに調整
    title_h = 42
    header_h = 40
    row_h = 58
    padding_x = 18
    padding_y = 16
    bottom_padding = 18
    table_w = 600
    table_h = title_h + header_h + row_h * len(visible_players) + padding_y * 2 + bottom_padding

    x0 = 20
    y0 = HEIGHT - table_h - 20
    x1 = x0 + table_w
    y1 = y0 + table_h

    # 半透明背景（以前より透明度を上げる = 少し薄くする）
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # 外枠
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)

    # タイトル
    title_y = y0 + padding_y + 24
    cv2.putText(frame, "MOLKKY SCORE", (x0 + padding_x, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    # 列位置
    col_name = x0 + 18
    col_score = x0 + 260
    box_w = 54
    box_h = 38
    box_gap = 16
    box_x0 = x0 + 360
    recent_box_x = [box_x0 + i * (box_w + box_gap) for i in range(3)]

    # ヘッダー
    header_base_y = y0 + padding_y + title_h + 24
    cv2.putText(frame, "PLAYER", (col_name, header_base_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    cv2.putText(frame, "TOTAL", (col_score, header_base_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    cv2.putText(frame, "LAST 3", (box_x0, header_base_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    line_y = header_base_y + 14
    cv2.line(frame, (x0 + 14, line_y), (x1 - 14, line_y), (180, 180, 180), 2)

    # プレイヤー行
    start_y = line_y + 36
    for i, player in enumerate(visible_players):
        row_center_y = start_y + i * row_h
        name = player.get("name", "")
        score = player.get("score", 0)
        recent_scores = player.get("recent_scores", ["", "", ""])[-3:]
        while len(recent_scores) < 3:
            recent_scores.insert(0, "")

        cv2.putText(frame, name, (col_name, row_center_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, str(score), (col_score, row_center_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        for j, s in enumerate(recent_scores):
            cell_x = recent_box_x[j]
            cell_y1 = row_center_y - box_h // 2
            cell_y2 = row_center_y + box_h // 2
            cv2.rectangle(frame, (cell_x, cell_y1), (cell_x + box_w, cell_y2), (70, 70, 70), -1)
            cv2.rectangle(frame, (cell_x, cell_y1), (cell_x + box_w, cell_y2), (170, 170, 170), 2)

            text = str(s)
            text_x = cell_x + 14
            if len(text) >= 2:
                text_x = cell_x + 10
            cv2.putText(frame, text, (text_x, row_center_y + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame


WINDOW_NAME = 'mac_pose_timeshift'
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

print("Starting live inference. Press 'q' to quit, 'ESC' or 'BackSpace' to return to Live.")
print(f"[INFO] スコア表示ファイル: {SCORE_FILE}")
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
    if is_recording and video_writer is not None:
        video_writer.write(display_frame)
                    
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
            
        cv2.rectangle(out_frame, (x1, 10), (x2, 50), bg_color, -1)
        cv2.rectangle(out_frame, (x1, 10), (x2, 50), (100, 100, 100), 2)
        
        if len(label) == 3:
            offset_x = 22
        elif len(label) == 4:
            offset_x = 16
        else:
            offset_x = 10
        cv2.putText(out_frame, label, (x1 + offset_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # 動作FPS表示
    cv2.putText(out_frame, f"FPS: {current_fps:.1f}", (WIDTH - 120, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
