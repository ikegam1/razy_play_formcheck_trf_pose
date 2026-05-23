import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import urllib.request
import os

# --- 解像度設定 ---
WIDTH = 1280
HEIGHT = 720

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

WINDOW_NAME = 'mac_pose_timeshift'
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

print("Starting live inference. Press 'q' to quit, 'ESC' or 'BackSpace' to return to Live.")
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
                    
    # 【録画】裏で常に最新映像（骨格描画済み）を保存
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
