# AI Hub

複数のAIプロバイダー（Google Gemini / OpenAI / Anthropic Claude 等）を単一のGradio WebGUIから操作できる統合プラットフォームです。
MEDフレームワーク（Memory Environment Distillation）の管理UIとしても機能します。

---

## 機能概要

| タブ | 機能 |
|------|------|
| **Playground** | チャットUI。プロバイダー・モデル・パラメータをGUIで切り替えてストリーミング推論 |
| **RAG** | テキスト / ファイルをFAISSに取り込み、ベクトル検索でクエリテスト |
| **Evals** | プロンプトセットを一括実行してスコアを比較 |
| **Training** | GRPOローカルトレーニング / Vertex AIジョブ送信 / Celeryジョブモニター |
| **Media** | 画像（Imagen / DALL-E）・動画（Veo）・音楽（MusicFX）生成 |
| **Settings** | APIキー設定・バックエンドヘルスチェック |

---

## アーキテクチャ

```
[Gradio WebGUI :7860]
        |
        | HTTP (BACKEND_URL)
        v
[FastAPI Backend :8000]
        |
  ┌─────┴──────┐
  │  Provider  │   Gemini / OpenAI / Anthropic / VoiceVox
  │  Plugin    │   Imagen / DALL-E / Veo / MusicFX
  └─────┬──────┘
        │
  ┌─────┴──────────┐
  │  Celery Worker │  ← Redis :6379
  │  (非同期ジョブ) │
  └────────────────┘

[VoiceVox Engine :50021]  (Docker sidecar)
```

---

## クイックスタート

### 1. 前提条件

- Docker Desktop（または Docker Engine + Compose Plugin）
- 使用するプロバイダーのAPIキー（最低1つ）

### 2. 環境変数の設定

```bash
cp .env.example .env
```

`.env` を編集してAPIキーを入力します：

```env
GEMINI_API_KEY=AIza...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Vertex AI や VoiceVox を使わない場合は空欄のままで起動可能です。

### 3. 起動

```bash
docker compose up --build
```

| サービス | URL |
|---------|-----|
| Gradio フロントエンド | http://localhost:7860 |
| FastAPI バックエンド | http://localhost:8000 |
| FastAPI ドキュメント (Swagger) | http://localhost:8000/docs |
| VoiceVox エンジン | http://localhost:50021 |

### 4. 停止

```bash
docker compose down
```

データ（FAISSインデックス等）を削除する場合：

```bash
docker compose down -v
```

---

## 各タブの使い方

### Playground

1. 右パネルで **Provider** と **Model** を選択
2. 必要に応じて **System Prompt** を入力
3. Temperature / Max Tokens / Top P を調整
4. メッセージを入力して **Send** または Enter

対応プロバイダー：

| Provider | 主なモデル |
|----------|-----------|
| `gemini` | gemini-2.0-flash, gemini-1.5-pro |
| `openai` | gpt-4o, gpt-4o-mini |
| `anthropic` | claude-opus-4-6, claude-sonnet-4-6 |

### RAG

1. **テキスト投入** — 本文を貼り付けてチャンクサイズを設定し「テキストを投入」
2. **ファイル投入** — `.txt` / `.md` ファイルをアップロード
3. **クエリテスト** — 検索クエリを入力して Top K 件を取得
4. 右パネルでドキュメント一覧の確認・削除が可能

### Training

#### GRPO (Local)

| 設定項目 | 説明 |
|---------|------|
| Model | HuggingFace Hub ID またはローカルパス |
| Dataset Path | JSONL ファイルパス（空欄で合成デモデータ使用）|
| Reward Model | `gemini`（Gemini APIで採点）または `rule_based` |
| Use LoRA | チェックを入れるとPEFT LoRAで軽量ファインチューニング |

「トレーニング開始」でCeleryキューに登録されます。

#### Vertex AI

> **重要**: `Max Cost (USD)` を必ず設定してください。GCPの請求アラートも合わせて設定することを推奨します。

必要項目：GCP Project ID、Staging Bucket（`gs://...`）、Container URI

#### Job Monitor

Celery Task UUID を入力してステータス確認、またはアクティブジョブ一覧を表示。

### Media

| タブ | Provider | 出力形式 |
|-----|---------|---------|
| Image | Imagen（Google）/ DALL-E（OpenAI）| 画像URL |
| Video | Veo（Google）| 動画URL（生成に数分かかる場合あり）|
| Music | MusicFX（Google）| 音声ファイル |

### Settings

- APIキー入力（ランタイム適用。ファイルには保存されません）
- 「Check Backend Health」でバックエンドの疎通確認

---

## OpenAI互換 API

バックエンドは `/v1/chat/completions` エンドポイントをOpenAI互換形式で提供します。

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash",
    "provider": "gemini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

`provider` フィールドで `gemini` / `openai` / `anthropic` を切り替えられます。
モデル名が `gemini-*` または `gpt-*` で始まる場合は自動判別します。

---

## プロンプトテンプレート

`templates/` ディレクトリに YAML 形式でテンプレートを追加できます。

```yaml
# templates/my_template.yaml
name: My Template
description: テンプレートの説明
system_prompt: |
  あなたは〇〇の専門家です。
user_template: "{{user_input}}"
tags:
  - custom
default_params:
  temperature: 0.5
  max_tokens: 1024
  top_p: 1.0
```

API: `GET /v1/templates` で一覧取得、`POST /v1/templates/{name}/render` でレンダリング。

---

## 環境変数 一覧

| 変数名 | 説明 | デフォルト |
|--------|------|----------|
| `GEMINI_API_KEY` | Google Gemini API キー | — |
| `OPENAI_API_KEY` | OpenAI API キー | — |
| `ANTHROPIC_API_KEY` | Anthropic API キー | — |
| `VOICEVOX_URL` | VoiceVox エンジンURL | `http://localhost:50021` |
| `GOOGLE_CLOUD_PROJECT` | GCP プロジェクトID | — |
| `GOOGLE_APPLICATION_CREDENTIALS` | サービスアカウントJSONパス | — |
| `REDIS_URL` | Celery ブローカーURL | `redis://localhost:6379/0` |
| `FAISS_INDEX_PATH` | FAISSインデックス保存先 | `./data/faiss_index` |
| `GCS_BUCKET` | GCS バケット名 | — |
| `BACKEND_URL` | フロントエンドからのバックエンドURL | `http://localhost:8000` |

---

## テスト実行

```bash
# Dockerで実行（推奨）
docker compose --profile test run --rm test

# ローカルで実行（仮想環境セットアップ済みの場合）
pip install -r requirements-test.txt
PYTHONPATH=. pytest tests/ -v --tb=short
```

---

## CI

GitHub Actions で全ブランチ・全PRに対して自動テストが実行されます（`.github/workflows/test.yml`）。

```
push / PR → Docker build (Dockerfile.test) → pytest tests/ -v → Pass/Fail
```

---

## ディレクトリ構成

```
ai-hub/
├── frontend/
│   ├── app.py                  # Gradio Blocks メインアプリ
│   └── tabs/
│       ├── playground.py       # チャットUI
│       ├── rag.py              # RAG管理
│       ├── evals.py            # Eval Dashboard
│       ├── training.py         # Training GUI
│       ├── media.py            # マルチモーダル生成
│       └── settings.py         # APIキー・設定
├── backend/
│   ├── main.py                 # FastAPI エントリーポイント
│   ├── api/                    # ルーター群
│   ├── providers/              # プロバイダープラグイン
│   │   ├── base.py             # 抽象基底クラス
│   │   ├── gemini.py
│   │   ├── openai_prov.py
│   │   ├── anthropic_prov.py
│   │   ├── voicevox.py
│   │   ├── imagen.py
│   │   ├── dalle.py
│   │   ├── veo.py
│   │   └── musicfx.py
│   ├── rag/                    # FAISSベクターストア
│   └── training/               # GRPO / LoRA / Vertex AI
├── templates/                  # プロンプトテンプレート (YAML)
├── tests/
├── docker-compose.yml
├── .env.example
└── CLAUDE.md
```

---

## 新しいプロバイダーの追加

1. `backend/providers/base.py` の `BaseProvider` を継承したクラスを作成
2. `generate` / `stream` / `get_models` / `validate_key` を実装
3. `backend/providers/__init__.py` の `PROVIDER_MAP` に登録
4. `frontend/utils.py` の `PROVIDER_MODEL_MAP` にモデル一覧を追加
