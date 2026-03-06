# パイプライン マルチモーダル化 実装計画

## 設計方針

Gemini・OpenAI・Anthropic の各プロバイダーはすでに `content` が `str` または OpenAI Vision 形式の `list` どちらでも処理できる実装になっている。**プロバイダー層の変更は最小限**で済み、主な変更はパイプライン層とフロントエンド層に集中する。

すべての変更は**テキストのみのパイプラインと完全な後方互換**を維持する。

---

## 変更対象ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `backend/api/pipeline.py` | スキーマ追加・エンドポイント修正 |
| `backend/pipeline/executor.py` | コンテキスト型拡張・テンプレート描画・メディア出力ラップ |
| `backend/providers/imagen.py` 他3件 | プロンプト抽出の防御的ガード追加 |
| `frontend/tabs/pipeline.py` | 画像入力UI・実行関数・結果表示の更新 |
| `frontend/utils.py` | `PROVIDER_MODEL_MAP` にメディアプロバイダーを追加 |

---

## Part 1 — スキーマ変更 (`backend/api/pipeline.py`)

### 1.1 `MultimodalContent` モデルを追加

```python
class MultimodalContent(BaseModel):
    type: str           # "text" | "image_url"
    text: str | None = None
    image_url: dict | None = None   # {"url": "data:image/png;base64,..."}
```

### 1.2 `PipelineRunRequest` に `input_parts` を追加

```python
class PipelineRunRequest(BaseModel):
    input: str = ""                                    # 既存・後方互換維持
    input_parts: list[MultimodalContent] | None = None # 新規追加
    definition: PipelineDefinition | None = None
```

`input_parts` が存在する場合は `input` より優先される。

### 1.3 `PipelineStep` に `step_type` を追加（任意）

```python
class PipelineStep(BaseModel):
    ...
    step_type: str = "text"  # "text" | "vision" | "image" | "video" | "audio"
```

エグゼキューターはプロバイダーの `modal_type` 属性から実行時に型を推論できるため、`step_type` は省略可能なヒントとして扱う。

### 1.4 エンドポイント修正

```python
effective_input: str | list = body.input
if body.input_parts:
    effective_input = [p.model_dump(exclude_none=True) for p in body.input_parts]
results = await run_pipeline(definition, effective_input)
```

---

## Part 2 — エグゼキューター変更 (`backend/pipeline/executor.py`)

### 2.1 コンテキスト型を拡張

```python
# Before
context: dict[str, str] = {"input": user_input}

# After
context: dict[str, str | list] = {"input": user_input}
```

### 2.2 `_content_to_text` ヘルパーを追加

```python
def _content_to_text(value: str | list) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(
        p.get("text", "") for p in value
        if isinstance(p, dict) and p.get("type") == "text"
    )
```

### 2.3 `_render` をマルチモーダル対応に更新

```python
def _render(template: str, context: dict[str, str | list]) -> str | list:
    # 単一プレースホルダーがリスト値を参照する場合はそのまま通す
    stripped = template.strip()
    m = re.fullmatch(r"\{\{([^}]+)\}\}", stripped)
    if m:
        key = m.group(1).strip()
        val = context.get(key)
        if isinstance(val, list):
            return val  # マルチパートコンテンツをそのまま通す

    # 通常の文字列レンダリング（リストはテキスト抽出して文字列化）
    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        val = context.get(key, match.group(0))
        return _content_to_text(val) if isinstance(val, list) else val

    return re.sub(r"\{\{([^}]+)\}\}", replace, template)
```

### 2.4 メディア生成ステップの出力をコンテンツリストにラップ

画像生成ステップの出力（URL/data URI）を次のVisionステップが受け取れるよう変換する：

```python
output = await provider.generate(messages, params)
result["output"] = output

# 画像生成プロバイダーの場合、出力を image_url コンテンツとしてラップ
if provider.modal_type == "image" and output:
    context_value: str | list = [
        {"type": "image_url", "image_url": {"url": output}}
    ]
else:
    context_value = output

context[f"step:{step_id}"] = context_value
context["prev_output"] = context_value
```

### 2.5 結果シリアライズ

```python
result["input"] = (
    rendered_input if isinstance(rendered_input, str)
    else json.dumps(rendered_input, ensure_ascii=False)
)
```

### 2.6 `run` シグネチャ変更

```python
async def run(self, definition: dict, user_input: str | list) -> list[dict]:
```

---

## Part 3 — メディアプロバイダー変更

対象: `imagen.py`, `dalle.py`, `veo.py`, `musicfx.py`

各プロバイダーの `generate()` でプロンプト抽出に防御的ガードを追加：

```python
# Before
prompt = messages[-1].get("content", "") if messages else ""

# After
raw = messages[-1].get("content", "") if messages else ""
prompt = raw if isinstance(raw, str) else _content_to_text(raw)
```

---

## Part 4 — フロントエンド変更 (`frontend/tabs/pipeline.py`)

### 4.1 画像入力UIを追加

```python
input_type_radio = gr.Radio(
    choices=["テキストのみ", "テキスト＋画像"],
    value="テキストのみ",
    label="入力タイプ",
)
user_input_box = gr.Textbox(
    label="入力テキスト",
    placeholder="パイプラインへの最初の入力を入力...",
    lines=6,
)
input_image_box = gr.Image(
    label="画像入力",
    type="base64",
    visible=False,
)

# テキスト＋画像選択時に画像入力を表示
input_type_radio.change(
    fn=lambda t: gr.update(visible=(t == "テキスト＋画像")),
    inputs=input_type_radio,
    outputs=input_image_box,
)
```

### 4.2 `_run` 関数を更新

```python
def _run(*all_args):
    user_input_text = all_args[-3]
    user_input_image = all_args[-2]
    input_type = all_args[-1]

    payload: dict = {"definition": defn}

    if input_type == "テキスト＋画像" and user_input_image:
        payload["input_parts"] = [
            {"type": "text", "text": str(user_input_text).strip()},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{user_input_image}"
            }},
        ]
        payload["input"] = ""
    else:
        payload["input"] = str(user_input_text).strip()
    ...
```

### 4.3 ステップ結果の画像レンダリング

```python
if r["output"].startswith("data:image/") or (
    r["output"].startswith("http") and any(
        ext in r["output"] for ext in [".png", ".jpg", ".webp"]
    )
):
    lines.append(f"![生成画像]({r['output']})")
else:
    lines.append(r["output"])
```

### 4.4 プロバイダードロップダウンにメディアプロバイダーを追加

```python
_BUILTIN_PROVIDERS = list(PROVIDER_MODEL_MAP.keys()) + ["imagen", "dalle", "veo", "musicfx"]
```

---

## Part 5 — `frontend/utils.py` の更新

`PROVIDER_MODEL_MAP` にメディアプロバイダーのモデル一覧を追加：

```python
PROVIDER_MODEL_MAP = {
    ...既存エントリ...
    "imagen": ["imagen-3.0-generate-001", "imagegeneration@006"],
    "dalle": ["dall-e-3", "dall-e-2"],
    "veo": ["veo-2.0-generate-001"],
    "musicfx": ["musicfx-001"],
}
```

---

## 後方互換チェックリスト

| 既存の動作 | 維持されるか | 理由 |
|---|---|---|
| `input: str` のみの `PipelineRunRequest` | ✅ | `input_parts` のデフォルトは `None` |
| 保存済みのテキストのみパイプライン | ✅ | `step_type` 不在でもデフォルト動作 |
| `context["prev_output"]` が文字列 | ✅ | `modal_type == "text"` のステップでは文字列のまま |
| Gradio フロントエンドの既存テキスト入力 | ✅ | `input_image_box` はデフォルト `None`、ペイロードは `input` にフォールバック |

---

## 実装順序

1. **`backend/pipeline/executor.py`** — `_content_to_text`, `_render` 更新, コンテキスト型拡張, メディア出力ラップ
2. **`backend/api/pipeline.py`** — `MultimodalContent` モデル追加, `input_parts` フィールド追加, エンドポイント修正
3. **メディアプロバイダー4件** — `_content_to_text` ガード追加
4. **`frontend/tabs/pipeline.py`** — 画像入力UI, `_run` 更新, 結果表示更新, プロバイダー拡張
5. **`frontend/utils.py`** — `PROVIDER_MODEL_MAP` 更新

手順 1〜3 はバックエンドのみで pytest で独立テスト可能。

---

## 対応するユースケース（完了後）

| ユースケース | 対応状況 |
|---|---|
| テキスト→テキスト パイプライン | ✅ 現状通り |
| 画像＋テキスト入力 → Vision モデル解析 | ✅ `input_parts` 経由 |
| 画像生成ステップ → Vision 解説ステップ | ✅ メディア出力の `image_url` ラップ経由 |
| テキスト → 画像生成（単独ステップ） | ✅ `imagen`/`dalle` をステップに指定 |
| 既存テキストのみパイプライン | ✅ 変更なしで動作継続 |
