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
frame_buffer = deque()       
is_playback = False          
playback_frames = []         
playback_index = 0           
playback_seconds = 0  

is_recording = False  
video_writer = None   
rec_start_time = 0.0  

# 実測FPS計測用のバッファ（直近30フレームの時間を記録）
fps_timestamps = deque(maxlen=30)
current_fps = 30.0

# UIボタン配置 (x1, x2, アクション種別, 設定値, ラベル)
BUTTONS = [
    (10, 110, "timeshift", 10, "< 10s"),
    (120, 220, "timeshift", 20, "< 20s"),
    (230, 330, "timeshift", 30, "< 30s"),
    (360, 460, "rec_start", None, "REC"),
    (470, 570, "rec_stop", None, "STOP")
]

# マウスクリックイベントの処理
def on_mouse_click(event, x, y, flags, param):
    global is_playback, playback_frames, playback_index, frame_buffer, playback_seconds
    global is_recording, video_writer, rec_start_time, current_fps
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= y <= 50:
            for x1, x2, btn_type, val, label in BUTTONS:
                if x1 <= x <= x2:
                    # 1. タイムシフト再生ボタン
                    if btn_type == "timeshift":
                        if not is_playback and len(frame_buffer) > 0:
                            print(f"\n[INFO] {val}秒前からの遅延再生を開始します...")
                            current_time = time.time()
                            target_time = current_time - val
                            playback_frames = [f for t, f in frame_buffer if t >= target_time]
                            playback_index = 0
                            playback_seconds = val
                            is_playback = True
                    
                    # 2. 録画開始ボタン
                    elif btn_type == "rec_start":
                        if not is_recording:
                            is_recording = True
                            rec_start_time = time.time()
                            filename = f"pose_rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                            
                            # 【修正】現在の「実測FPS」をそのまま動画のフレームレートとして採用
                            # 異常値対策として下限10、上限60の安全ガードを設置
                            video_fps = max(10.0, min(60.0, current_fps))
                            
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(filename, fourcc, video_fps, (WIDTH, HEIGHT))
                            print(f"\n[INFO] 録画を開始しました (設定FPS: {video_fps:.1f}): {filename}")
                    
                    # 3. 録画停止ボタン
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

print("Starting live inference. Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できません。")
        break
        
    current_time = time.time()
    
    # 【新機能】実測FPSの計算（直近30コマの処理速度からリアルタイム算出）
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
                    
    # 録画処理：実測FPSと同じペースで「最新のライブ映像」を裏で書き込み続ける
    if is_recording and video_writer is not None:
        video_writer.write(display_frame)
                    
    # 過去30秒のライブフレームを常時バッファにストック
    frame_buffer.append((current_time, display_frame.copy()))
    while frame_buffer and (current_time - frame_buffer[0][0] > 30.0):
        frame_buffer.popleft()
        
    # メイン画面に出力するベースフレームの決定
    if is_playback:
        if playback_index < len(playback_frames):
            out_frame = playback_frames[playback_index].copy()
            playback_index += 1
            cv2.putText(out_frame, f"-{playback_seconds}sec PLAYBACK", (WIDTH - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            print("[INFO] 遅延再生が終了しました。ライブに戻ります。")
            is_playback = False
            out_frame = display_frame.copy()
    else:
        out_frame = display_frame.copy()
        
    # 録画時間表示ロジック
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
        elif btn_type == "timeshift" and is_playback and val == playback_seconds:
            bg_color = (0, 165, 255)     
            text_color = (255, 255, 255)
        else:
            bg_color = (220, 220, 220)   
            text_color = (0, 0, 0)
            
        cv2.rectangle(out_frame, (x1, 10), (x2, 50), bg_color, -1)
        cv2.rectangle(out_frame, (x1, 10), (x2, 50), (100, 100, 100), 2)
        
        offset_x = 12 if "s" in label or "STOP" in label else 25
        cv2.putText(out_frame, label, (x1 + offset_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    # 【新機能】現在の動作FPSを画面右下にさりげなく表示
    cv2.putText(out_frame, f"FPS: {current_fps:.1f}", (WIDTH - 120, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, out_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
