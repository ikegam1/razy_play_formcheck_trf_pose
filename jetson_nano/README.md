# trt_pose with Timeshift Playback

Jetson Nano上でTensorRTを用いて高速に動作するリアルタイム骨格推定（`trt_pose`）システムです。
標準のリアルタイム推論機能に加え、画面上のボタンをワンクリックするだけで、過去「10秒」「20秒」「30秒」の推論映像をその場で巻き戻して等倍再生（タイムシフト再生）できる機能を搭載しています。スポーツのフォームチェックや動作確認に最適です。

## 特徴
- **TensorRT 8.x 最適化**: Jetson Nano（JetPack 4.6）環境で高効率なFP16推論を実行。
- **オンメモリ・リングバッファ**: Jetsonの限られたメモリ（4GB）を枯渇させないスマートなメモリ管理。
- **バックグラウンド推論継続**: 過去の映像をプレイバックしている間も、裏ではカメラ映像のキャプチャと推論を継続するため、ライブ映像へシームレスに復帰可能。

## 動作環境
- **Hardware**: Jetson Nano (Developer Kit)
- **OS**: JetPack 4.6 (Ubuntu 18.04 LTS)
- **Python**: 3.6.9
- **TensorRT**: 8.0.x
- **CUDA**: 10.2
- **Camera**: USB Web Camera (`/dev/video1` 準拠)

## セットアップ手順

### 1. 依存ライブラリのインストール（Jetson公式準拠）
Jetson環境の PyTorch および torchvision は、NVIDIA公式のインストール手順に従って導入してください。

### 2. torch2trt のインストール (重要)
JetPack 4.6 (TensorRT 8.x) 環境で安定動作する `v0.4.0` を指定してビルドします。
:::bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
git checkout v0.4.0
sudo python3 setup.py clean --all
sudo python3 setup.py install --plugins
:::

### 3. trt_pose のインストール
:::bash
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
:::

### 4. 当リポジトリのクローンと外部ファイルの準備
当リポジトリをクローンしたのち、`trt_pose` 公式リポジトリ等からモデルの重み（`.pth`）およびタスク設定ファイル（`.json`）をスクリプトと同じディレクトリに配置してください。

- `human_pose.json`
- `resnet18_baseline_att_224x224_A_epoch_249.pth`

:::bash
git clone <あなたのGitHubリポジトリURL>
cd <リポジトリ名>/tasks/human_pose
pip3 install -r requirements.txt
:::

## 使い方

スクリプトを実行します。USBカメラは `/dev/video1` として認識されている必要があります。
:::bash
python3 run_webcam_trt.py
:::

### 画面操作
- **初回起動時**: TensorRTのモデル最適化（コンパイル）が走るため、起動までに3〜5分かかります。2回目以降はキャッシュがロードされ、一瞬で起動します。
- **`< 10s` `/` `< 20s` `/` `< 30s` ボタン**: 画面左上のボタンをマウスで左クリックすると、それぞれの秒数だけ過去に遡って骨格推定付きの映像をプレイバックします。
- **`q` キー**: アプリケーションを終了します。
