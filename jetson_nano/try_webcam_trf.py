import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import os
import time
from collections import deque

# 1. タスク設定（骨格定義）のロード
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# 2. 基本モデルの構築と重みの読み込み
print("Loading PyTorch model...")
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

# 3. TensorRTへの最適化（初回のみ数分かかります。2回目以降はキャッシュをロード）
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
data = torch.zeros((1, 3, 224, 224)).cuda()

if not os.path.exists(OPTIMIZED_MODEL):
    print("Optimizing model with TensorRT... This will take a few minutes (First time only).")
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    print("Optimization complete!")
else:
    print("Loading optimized TensorRT model from cache...")
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

# 4. カメラ設定 (/dev/video1 を指定)
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    x = transforms.functional.to_tensor(image).to(device)
    x.sub_(mean[:, None, None]).div_(std[:, None, None])
    return x[None, ...]

# --- タイムシフト（遅延再生）用の変数定義 ---
frame_buffer = deque()       # 過去30秒の (タイムスタンプ, フレーム) を保存するバッファ
is_playback = False          # 現在遅延再生中かどうかのフラグ
playback_frames = []         # 再生対象のフレームリスト
playback_index = 0           # 現在再生しているフレームのインデックス

# ボタンの位置定義 (x1, x2, 秒数, ラベル)
BUTTONS = [
    (10, 110, 10, "< 10s"),
    (120, 220, 20, "< 20s"),
    (230, 330, 30, "< 30s")
]

# マウスクリックイベントの処理
def on_mouse_click(event, x, y, flags, param):
    global is_playback, playback_frames, playback_index, frame_buffer
    if event == cv2.EVENT_LBUTTONDOWN:
        # y軸がボタンの高さ範囲内（10px〜50px）かチェック
        if 10 <= y <= 50:
            for x1, x2, seconds, label in BUTTONS:
                # どのボタンの横幅に収まっているか判定
                if x1 <= x <= x2:
                    if not is_playback and len(frame_buffer) > 0:
                        print(f"\n[INFO] {seconds}秒前からの遅延再生を開始します...")
                        current_time = time.time()
                        target_time = current_time - seconds
                        
                        # 指定した秒数より新しいフレームだけをバッファから抽出
                        playback_frames = [f for t, f in frame_buffer if t >= target_time]
                        playback_index = 0
                        is_playback = True
                    break

# ウィンドウを作成してマウスイベントを紐付け
WINDOW_NAME = 'trt_pose'
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse_click)

print("Starting live inference. Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    current_time = time.time()
    
    # モデルの入力サイズ(224x224)にリサイズ
    frame_resized = cv2.resize(frame, (224, 224))
    x = preprocess(frame_resized)
    
    # 推論実行
    with torch.no_grad():
        cmap, paf = model_trt(x)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    
    # 骨格のパースと描画
    counts, objects, peaks = parse_objects(cmap, paf)
    draw_objects(frame_resized, counts, objects, peaks)
    
    # 画面に見やすいよう、少し大きくリサイズして表示用ベースを作成
    display_frame = cv2.resize(frame_resized, (640, 480))
    
    # 1. 現在のリアルタイム推論フレームをバッファに常時追加
    frame_buffer.append((current_time, display_frame.copy()))
    
    # 2. 最大秒数（30秒）以上古いフレームを自動的に破棄
    while frame_buffer and (current_time - frame_buffer[0][0] > 30.0):
        frame_buffer.popleft()
        
    # 3. 画面に表示するフレームを決定（ライブ映像か、遅延再生映像か）
    if is_playback:
        if playback_index < len(playback_frames):
            out_frame = playback_frames[playback_index].copy()
            playback_index += 1
            # 再生中であることを示す赤いインジケータを右上に表示
            cv2.putText(out_frame, "● PLAYBACK", (460, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            print("[INFO] 遅延再生が終了しました。ライブに戻ります。")
            is_playback = False
            out_frame = display_frame.copy()
    else:
        out_frame = display_frame.copy()
        
    # 4. 「10s」「20s」「30s」ボタンを画面左上に描画
    for x1, x2, seconds, label in BUTTONS:
        # ボタンの背景（薄いグレーの矩形）
        cv2.rectangle(out_frame, (x1, 10), (x2, 50), (220, 220, 220), -1)
        # ボタンの枠線
        cv2.rectangle(out_frame, (x1, 10), (x2, 50), (100, 100, 100), 2)
        # ボタンのテキスト
        cv2.putText(out_frame, label, (x1 + 12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 画面表示
    cv2.imshow(WINDOW_NAME, out_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
