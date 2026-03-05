# AI Hub — Claude Code Handoff Document

## プロジェクト概要

AI Hubは、主要AIスタジオ（Google AI Studio / OpenAI / Anthropic等）の共通機能を抽象化・テンプレート化し、単一のGradio WebGUIから統合操作できるプラットフォーム。  
MEDフレームワーク（Memory Environment Distillation）の開発・管理UIとしても機能する。

---

## 現在のステータス

- [x] プロジェクト計画策定済み
- [ ] Phase 1: コア基盤実装
- [ ] Phase 2: 拡張機能
- [ ] Phase 3: トレーニングGUI
- [ ] Phase 4: マルチモーダル

**次にやること（Phase 1 開始タスク）:**
1. `backend/` の FastAPI skeleton を作成
2. `providers/base.py` の抽象基底クラスを定義
3. `providers/gemini.py` を最初のプラグインとして実装
4. `frontend/app.py` に Gradio Playground タブを実装

---

## ディレクトリ構成（目標）

```
ai-hub/
├── frontend/
│   ├── app.py              # Gradio メインアプリ (Blocks API)
│   ├── tabs/
│   │   ├── playground.py   # Prompt Playground タブ
│   │   ├── evals.py        # Eval Dashboard タブ
│   │   ├── training.py     # Training GUI タブ
│   │   ├── rag.py          # RAG管理タブ
│   │   └── settings.py     # APIキー・設定タブ
│   └── themes/
│       └── hub_theme.py    # カスタムGradioテーマ
├── backend/
│   ├── main.py             # FastAPI エントリーポイント
│   ├── api/
│   │   ├── chat.py         # /v1/chat/completions (OpenAI互換)
│   │   ├── models.py       # /v1/models
│   │   └── jobs.py         # /v1/training/jobs
│   ├── providers/
│   │   ├── base.py         # 抽象基底クラス (必ず継承すること)
│   │   ├── gemini.py       # Google Gemini
│   │   ├── openai_prov.py  # OpenAI GPT-4o
│   │   ├── anthropic_prov.py # Claude
│   │   └── voicevox.py     # VoiceVox TTS (Speachesフォーク統合)
│   ├── training/
│   │   ├── grpo.py         # GRPOトレーニングループ
│   │   ├── lora.py         # LoRA / TinyLoRA 適用
│   │   └── vertex_job.py   # Vertex AI Custom Job 送信
│   └── rag/
│       ├── faiss_store.py  # FAISSベクターストア管理
│       └── retriever.py    # RAG検索ロジック
├── templates/              # プロンプトテンプレート (YAML形式)
├── tests/
│   ├── test_providers.py   # プロバイダーモックテスト
│   └── test_api.py
├── docker-compose.yml      # VoiceVox等のサイドカー含む
├── requirements.txt
├── .env.example
└── CLAUDE.md               # このファイル
```

---

## アーキテクチャ原則

### レイヤー構成

```
[Gradio WebGUI]  ←→  [FastAPI Backend]
       ↓                      ↓
  Playground UI         Plugin Manager
  Eval Dashboard        Provider Abstraction Layer
  Training GUI          Template Engine
       ↓                      ↓
         [Provider Plugins]
         Gemini | GPT-4o | Claude | VoiceVox | ...
                      ↓
         [Training Backend]       [Storage]
         Vertex AI / Local        FAISS / GCS
```

### 重要な設計制約

- **プロバイダーはすべて `base.py` を継承**すること。直接APIを叩くコードをタブ層に書かない
- **フロントエンドにAPIキーを渡さない**。キーはバックエンドの環境変数のみ
- **OpenAI互換エンドポイント**を維持する（`/v1/chat/completions`）。他ツールとの連携のため
- **トレーニングジョブはキュー経由**（Celery + Redis）。Gradioのイベントループをブロックしない

---

## providers/base.py — 実装すべきインターフェース

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseProvider(ABC):
    modal_type: str = "text"  # "text" | "image" | "audio" | "video"
    
    @abstractmethod
    async def generate(self, messages: list, params: dict) -> str:
        """非ストリーミング推論"""
        pass

    @abstractmethod
    async def stream(self, messages: list, params: dict) -> AsyncGenerator[str, None]:
        """ストリーミング推論 (SSE)"""
        pass

    @abstractmethod
    def get_models(self) -> list[str]:
        """利用可能なモデル一覧"""
        pass

    @abstractmethod
    def validate_key(self) -> bool:
        """APIキーの有効性確認"""
        pass
```

---

## MEDフレームワーク連携

AI HubはMEDの操作UIとして機能する。対応関係は以下の通り。

| MEDコンポーネント | AI Hub対応箇所 | 実装メモ |
|---|---|---|
| Teacher (Gemini API) | `providers/gemini.py` | 報酬生成・データキュレーションに使用 |
| Student (7-8B OSS) | `training/` タブ + HuggingFace Hub | ローカルパスまたはHFモデルIDを指定 |
| FAISS Vector Store | `rag/faiss_store.py` + RAGタブ | MEDのFAISSと共有インスタンス可 |
| GRPOトレーニング | `training/grpo.py` + Vertex Jobタブ | カスタムコンテナでVertex AIに送信 |
| Model Router | Settings タブのルーティング設定 | クエリ複雑度閾値をGUI上で設定 |

---

## VoiceVox統合メモ

`providers/voicevox.py` はSpeachesフォークプロジェクトの実装を移植する。

- **Docker Compose** に VoiceVox エンジンをサイドカーとして追加済み想定
- **WAV → MP3変換** は ffmpeg 経由（`subprocess` または `ffmpeg-python`）
- **スピーカー一覧** は VoiceVox エンジンの `/speakers` エンドポイントを動的に取得
- エンドポイントURL: `http://voicevox:50021`（Docker内部ネットワーク）

---

## 技術スタック

| レイヤー | 技術 | バージョン目安 |
|---|---|---|
| フロントエンド | Gradio (Blocks API) | 5.x |
| バックエンド | FastAPI + Uvicorn | 0.110+ |
| ジョブキュー | Celery + Redis | - |
| ベクターDB | FAISS + LangChain | - |
| コンテナ | Docker Compose | v2 |
| トレーニング | HuggingFace PEFT + Vertex AI | - |
| テスト | pytest | - |

---

## フェーズ別タスク

### Phase 1（Week 1-2）— コア基盤 ← 現在ここから開始

- [ ] `requirements.txt` を作成
- [ ] `backend/main.py` FastAPI skeleton
- [ ] `backend/providers/base.py` 抽象クラス
- [ ] `backend/providers/gemini.py` 実装
- [ ] `backend/providers/openai_prov.py` 実装
- [ ] `frontend/app.py` Gradio Blocks ベース
- [ ] `frontend/tabs/playground.py` — チャットUI + パラメータパネル
- [ ] `frontend/tabs/settings.py` — APIキー入力UI
- [ ] `docker-compose.yml` 基本構成
- [ ] `tests/test_providers.py` モックテスト

### Phase 2（Week 3-4）— 拡張機能

- [ ] `providers/voicevox.py` TTS統合
- [ ] `providers/anthropic_prov.py`
- [ ] `rag/faiss_store.py` + RAGタブ
- [ ] `templates/` プロンプトライブラリUI
- [ ] `tabs/evals.py` Eval Dashboard

### Phase 3（Week 5-6）— トレーニングGUI

- [ ] `training/grpo.py` GRPOループ
- [ ] `training/lora.py` LoRA/TinyLoRA
- [ ] `training/vertex_job.py` Vertex AI送信
- [ ] `tabs/training.py` Training GUI
- [ ] Celery + Redis ジョブキュー統合

### Phase 4（Week 7-8）— マルチモーダル

- [ ] 画像プロバイダー（Imagen / DALL-E）
- [ ] 動画プロバイダー（Veo / Runway）
- [ ] 音楽プロバイダー（MusicFX）
- [ ] E2Eテスト・ドキュメント整備

---

## 環境変数（.env.example）

```env
# LLM Providers
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# VoiceVox
VOICEVOX_URL=http://localhost:50021

# Vertex AI
GOOGLE_CLOUD_PROJECT=
GOOGLE_APPLICATION_CREDENTIALS=

# Redis (Celery)
REDIS_URL=redis://localhost:6379/0

# Storage
FAISS_INDEX_PATH=./data/faiss_index
GCS_BUCKET=
```

---

## 既知の制約・注意事項

- Google AI Studio上での直接トレーニング拡張は**不可**。実行はVertex AIまたはローカルに委譲
- Geminiの重み・内部アーキテクチャへのアクセスは**不可**（クローズド）
- Gradioはスケーラビリティに限界あり。Phase4以降でNext.js移行を検討
- Vertex AI Jobにはコスト上限設定を**必ず**入れること

---

## 参考リンク

- [Gradio Blocks API](https://www.gradio.app/docs/gradio/blocks)
- [FastAPI](https://fastapi.tiangolo.com/)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [Vertex AI Custom Training](https://cloud.google.com/vertex-ai/docs/training/overview)
- [VoiceVox Engine API](https://voicevox.github.io/voicevox_engine/api/)
