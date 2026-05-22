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
```bash
git clone [https://github.com/NVIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
cd torch2trt
git checkout v0.4.0
sudo python3 setup.py clean --all
sudo python3 setup.py install --plugins
