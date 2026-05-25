# Mölkky Pose Replay & Live Score Overlay

Mölkky の投擲フォーム確認・遅延再生・録画・スコア表示・HLS配信を行うための Python アプリ一式です。

このリポジトリは大きく分けて、以下の3つで構成します。

1. **映像アプリ**  
   カメラ映像を取得し、MediaPipeで骨格推定を行い、遅延再生・録画・スコアオーバーレイ・HLS出力を行います。

2. **スコア入力アプリ**  
   Mölkky のスコアを入力し、`scores.json` を更新します。映像アプリはこの `scores.json` を読み込んで、画面上にスコア表を表示します。

3. **HLS配信用ファイル**  
   ローカルで生成された HLS ファイルを Cloudflare R2 にアップロードし、Cloudflare Pages 上のWebページで再生します。

---

## ファイル構成

```text
.
├── molkky_pose_timeshift_with_score_v10.py   # 映像・骨格推定・遅延再生・HLS生成アプリ
├── molkky_score_input_mac_cv2_v7.py          # Mac用スコア入力アプリ
├── upload_hls_to_r2_dvr_v2.sh                # Cloudflare R2アップロード用スクリプト
├── index_hls_dvr_controls.html               # Cloudflare Pages用 視聴ページ
├── scores.json                               # スコア共有ファイル。スコアアプリが生成・更新
├── hls_output/                               # HLS出力先。映像アプリが生成
└── .uploaded_hls_segments.txt                # R2アップロード済みセグメント管理ファイル
```

最終的に GitHub に置く場合は、ローカル生成物をコミットしないようにしてください。

推奨 `.gitignore` は以下です。

```gitignore
__pycache__/
*.pyc

scores.json
hls_output/
.uploaded_hls_segments.txt
hls_playlist_snapshots/

*.mp4
*.mov
*.avi
.DS_Store
```

---

## 動作環境

主に macOS を想定しています。

### 必要なもの

- Python 3.10 以上
- OpenCV
- MediaPipe
- NumPy
- Pillow
- ffmpeg
- Node.js / npm
- Wrangler
- Cloudflare R2
- Cloudflare Pages

### Pythonパッケージ

```bash
python3 -m pip install opencv-python mediapipe numpy pillow
```

### ffmpeg

```bash
brew install ffmpeg
```

### Wrangler

```bash
npm install -g wrangler
wrangler login
```

---

## 1. 映像アプリ

### 起動

```bash
python3 molkky_pose_timeshift_with_score_v10.py
```

### 主な機能

- カメラ映像取得
- MediaPipeによる骨格推定
- 遅延再生
- 録画
- `scores.json` の読み込み
- スコア表の右上表示
- HLSファイル生成
- 終了時のHLSローカルファイル掃除

### HLS出力

映像アプリを起動すると、以下にHLSファイルが生成されます。

```text
hls_output/
├── stream.m3u8
├── stream_00000.ts
├── stream_00001.ts
└── ...
```

`v10` では安定性を優先し、HLS出力は以下の設定になっています。

```python
HLS_FPS = 15
HLS_SEGMENT_SECONDS = 4
HLS_LIST_SIZE = 10
```

骨格推定などで処理負荷が高い場合、30fpsでHLSを生成すると実時間より速い動画として扱われることがあります。  
そのため、HLS用の専用スレッドで最新フレームを一定間隔でffmpegへ送り、動画の時間軸が実時間に近くなるようにしています。

### 終了時の掃除

映像アプリ終了時に以下を削除します。

```text
.uploaded_hls_segments.txt
hls_output/*
```

これにより、次回起動時に古いHLSセグメントやアップロード済みログが残ることによる不整合を避けます。

---

## 2. スコア入力アプリ

### 起動

```bash
python3 molkky_score_input_mac_cv2_v7.py
```

### 起動時入力

起動すると、ターミナル上で以下を入力します。

- プレイヤー数
- プレイヤー名
- 投げ順モード
  - `Reverse`
  - `Slide`

### 操作

画面上のボタン、またはキーボードで操作できます。

| 操作 | キー |
|---|---|
| 0〜9点 | `0`〜`9` |
| 10点 | `a` |
| 11点 | `b` |
| 12点 | `c` |
| Undo | `u` |
| Reset | `r` |
| Burst | `x` |
| Quit | `q` |

### スコアルール

- 0点はミスとして `-` 表示
- 0点3回でOUT
- OUTになったプレイヤーは以後スキップ
- 1人以外がOUTになった場合、生き残りを50点としてセット終了
- 50点ちょうどでセット終了
- 50点を超えた場合はバーストし、25点へ戻る
- バースト時の直近投には `B` ではなく、入力された数字を表示
- セット終了時、勝者のセットカウントを+1
- Reset、アプリ起動時、終了時はセットカウントを0に戻す

### 投げ順

#### Reverse

セット終了ごとに投げ順が逆になります。

```text
ABC → CBA → ABC
```

#### Slide

セット終了ごとに先頭プレイヤーがずれます。

```text
ABC → BCA → CAB
```

---

## 3. scores.json

映像アプリとスコア入力アプリは、同じディレクトリの `scores.json` で連携します。

例:

```json
{
  "players": [
    {
      "name": "Player 1",
      "score": 24,
      "recent_scores": [8, 10, 6],
      "set_count": 0,
      "miss_count": 0,
      "is_out": false
    },
    {
      "name": "Player 2",
      "score": 18,
      "recent_scores": ["-", 8, 10],
      "set_count": 1,
      "miss_count": 1,
      "is_out": false
    }
  ],
  "meta": {
    "current_player": "Player 1",
    "set_number": 1,
    "order_mode": "Reverse"
  }
}
```

注意点:

- ミス表記の `-` はJSON上では文字列として `"-"` と書きます。
- `recent_scores` は新しい投擲が左側に来ます。
  - 例: `2, 8, 8`
- 映像アプリ側では `miss_count` に応じてプレイヤー名横に赤い `X` を表示します。
- `is_out` が `true` の場合、スコア欄は `OUT` と表示されます。

---

## 4. Cloudflare R2 配信

### R2バケット

例:

```text
molkky-score-hls
```

### 公開URL

例:

```text
https://pub-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.r2.dev
```

このREADME内のサンプルでは以下のURLを使っています。

```text
https://pub-9d9710ab45014df8b3db7b6fa45b8307.r2.dev
```

自分の環境に合わせて、以下のファイル内のURLを書き換えてください。

- `upload_hls_to_r2_dvr_v2.sh`
- `index_hls_dvr_controls.html`

### CORS設定

Cloudflare Pages から R2 の `.m3u8` / `.ts` を読むため、CORSを設定します。

`cors.json`:

```json
{
  "rules": [
    {
      "allowed": {
        "origins": ["*"],
        "methods": ["GET", "HEAD"]
      }
    }
  ]
}
```

適用:

```bash
wrangler r2 bucket cors set molkky-score-hls --file cors.json
wrangler r2 bucket cors list molkky-score-hls
```

本番では `origins` を Cloudflare Pages のURLや独自ドメインに絞ることを推奨します。

---

## 5. HLSアップロード

### 起動

映像アプリとは別ターミナルで起動します。

```bash
./upload_hls_to_r2_dvr_v2.sh
```

### 役割

- `hls_output/*.ts` をR2へアップロード
- R2上に実際に存在するか検証
- DVR形式の長めの `stream.m3u8` を生成
- `stream.m3u8` をR2へアップロード
- シークバーで戻れる範囲を確保

### 主な設定

`upload_hls_to_r2_dvr_v2.sh` 内で調整します。

```bash
PUBLISH_DELAY_SEC=12
DVR_WINDOW_SEC=180
SEGMENT_DURATION_SEC=4
```

| 項目 | 説明 |
|---|---|
| `PUBLISH_DELAY_SEC` | ライブより何秒遅らせて公開するか |
| `DVR_WINDOW_SEC` | シークバーで戻れる時間 |
| `SEGMENT_DURATION_SEC` | HLSセグメント長。映像アプリ側と合わせる |

`molkky_pose_timeshift_with_score_v10.py` では `HLS_SEGMENT_SECONDS = 4` なので、`SEGMENT_DURATION_SEC=4` にしてください。

### 直接確認

```bash
curl https://pub-9d9710ab45014df8b3db7b6fa45b8307.r2.dev/live/stream.m3u8
```

`.m3u8` の中に複数の `.ts` が表示されていればOKです。

---

## 6. Cloudflare Pages

`index_hls_dvr_controls.html` を `index.html` として Cloudflare Pages にアップロードします。

### Pages用ファイル

```text
index.html
```

中のHLS URLを自分のR2 URLに合わせます。

```javascript
const src = "https://pub-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.r2.dev/live/stream.m3u8";
```

### Web画面の操作

- 再生 / 停止
- シークバー
- `-30秒`
- `-10秒`
- `+10秒`
- `最新へ`
- 最新から何秒遅れているかの表示

---

## 7. 推奨起動手順

ターミナルを3つ使います。

### ターミナル1: 映像アプリ

```bash
python3 molkky_pose_timeshift_with_score_v10.py
```

### ターミナル2: スコア入力アプリ

```bash
python3 molkky_score_input_mac_cv2_v7.py
```

### ターミナル3: R2アップロード

```bash
./upload_hls_to_r2_dvr_v2.sh
```

Cloudflare Pages のURLを開くと、R2上のHLSを再生できます。

---

## 8. トラブルシュート

### `scores.json` の読み込みに失敗する

`recent_scores` にミスを入れる場合は、必ず文字列にしてください。

NG:

```json
"recent_scores": [8, -, 6]
```

OK:

```json
"recent_scores": [8, "-", 6]
```

確認:

```bash
python3 -m json.tool scores.json
```

---

### 日本語が `????` になる

Pillowが必要です。

```bash
python3 -m pip install pillow
```

macOS上の日本語フォントを自動検出して描画します。

---

### HLSが生成されない

ffmpegが入っているか確認します。

```bash
which ffmpeg
ffmpeg -version
```

未インストールの場合:

```bash
brew install ffmpeg
```

---

### R2へアップロードしたのにbucketに見えない

Wranglerのログに以下が出ている場合、ローカル環境へアップロードしています。

```text
Resource location: local
Use --remote if you want to access the remote instance.
```

必ず `--remote` を付けてください。

```bash
wrangler r2 object put molkky-score-hls/live/stream.m3u8 \
  --file hls_output/stream.m3u8 \
  --content-type application/vnd.apple.mpegurl \
  --remote
```

`upload_hls_to_r2_dvr_v2.sh` では `--remote` を指定済みです。

---

### `.ts` が404になる

アップロードログとR2の実体がずれている可能性があります。

```bash
rm -f .uploaded_hls_segments.txt
./upload_hls_to_r2_dvr_v2.sh
```

`v10` の映像アプリでは終了時に `.uploaded_hls_segments.txt` と `hls_output/*` を削除します。

---

### 再生が速い / 倍速に見える

ブラウザの `playbackRate` ではなく、HLS生成時のFPSタイムベースが原因のことがあります。

`v10` では以下の対策を入れています。

- HLS出力を15fpsに制限
- HLS専用スレッドで一定間隔出力
- 最新フレームを複製して実時間に合わせる
- 4秒セグメントでアップロード負荷を軽減

---

### 再生が固まりやすい

以下を調整してください。

`upload_hls_to_r2_dvr_v2.sh`:

```bash
PUBLISH_DELAY_SEC=16
DVR_WINDOW_SEC=180
```

安定優先なら `PUBLISH_DELAY_SEC` を大きくします。  
遅延を減らしたい場合は、安定確認後に少しずつ下げます。

---

## 9. 注意事項

- R2のPublic Development URLはテスト用途です。本番運用ではCustom Domainの利用を推奨します。
- HLS配信では `.ts` が増え続けるため、長時間運用する場合は古い `.ts` の削除処理を追加してください。
- `scores.json` はスコア入力アプリがatomic writeで更新します。
- HLSの安定性は、MacのCPU負荷、ネットワーク、R2アップロード速度に影響されます。

---

## 10. ライセンス

必要に応じて `LICENSE` ファイルを追加してください。
