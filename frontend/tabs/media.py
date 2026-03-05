"""Multimodal media generation tab — image / video / music."""
from __future__ import annotations

import os

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ======================================================================
# Backend helpers
# ======================================================================

def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    try:
        resp = httpx.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Action functions
# ======================================================================

def generate_image(
    prompt: str, provider: str, size: str, aspect_ratio: str, quality: str
) -> tuple[str | None, str]:
    if not prompt.strip():
        return None, "プロンプトを入力してください"
    payload = {
        "prompt": prompt,
        "provider": provider,
        "size": size,
        "aspect_ratio": aspect_ratio,
        "quality": quality,
    }
    result = _post("/v1/media/images/generate", payload)
    if "error" in result:
        return None, f"Error: {result['error']}"
    url = result.get("url", "")
    if not url:
        return None, "画像が生成されませんでした"
    return url, f"完了 ({result.get('provider', provider)})"


def generate_video(
    prompt: str, provider: str, duration: int, aspect_ratio: str
) -> tuple[str | None, str]:
    if not prompt.strip():
        return None, "プロンプトを入力してください"
    payload = {
        "prompt": prompt,
        "provider": provider,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
    }
    result = _post("/v1/media/videos/generate", payload, timeout=600)
    if "error" in result:
        return None, f"Error: {result['error']}"
    url = result.get("url", "")
    if not url:
        return None, "動画が生成されませんでした"
    return url, f"完了 ({result.get('provider', provider)})"


def generate_music(prompt: str, provider: str, duration: int) -> tuple[str | None, str]:
    if not prompt.strip():
        return None, "プロンプトを入力してください"
    payload = {
        "prompt": prompt,
        "provider": provider,
        "duration": int(duration),
    }
    result = _post("/v1/media/music/generate", payload, timeout=180)
    if "error" in result:
        return None, f"Error: {result['error']}"
    url = result.get("url", "")
    if not url:
        return None, "音楽が生成されませんでした"
    return url, f"完了 ({result.get('provider', provider)})"


# ======================================================================
# Tab builder
# ======================================================================

def build_media_tab() -> gr.Tab:
    with gr.Tab("Media") as tab:
        gr.Markdown("## マルチモーダル生成")

        with gr.Tabs():
            # ----------------------------------------------------------
            # Image generation
            # ----------------------------------------------------------
            with gr.Tab("Image"):
                gr.Markdown("### 画像生成")
                with gr.Row():
                    with gr.Column(scale=2):
                        img_prompt = gr.Textbox(
                            label="プロンプト",
                            placeholder="A serene mountain landscape at sunset...",
                            lines=3,
                        )
                        with gr.Row():
                            img_provider = gr.Dropdown(
                                label="Provider",
                                choices=["imagen", "dalle"],
                                value="imagen",
                            )
                            img_size = gr.Dropdown(
                                label="Size (DALL-E)",
                                choices=["1024x1024", "1024x1792", "1792x1024"],
                                value="1024x1024",
                            )
                            img_aspect = gr.Dropdown(
                                label="Aspect Ratio (Imagen)",
                                choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                                value="1:1",
                            )
                            img_quality = gr.Dropdown(
                                label="Quality (DALL-E)",
                                choices=["standard", "hd"],
                                value="standard",
                            )
                        img_btn = gr.Button("画像生成", variant="primary")
                        img_status = gr.Textbox(label="ステータス", interactive=False)

                    with gr.Column(scale=2):
                        img_output = gr.Image(label="生成結果", type="filepath")

                img_btn.click(
                    fn=generate_image,
                    inputs=[img_prompt, img_provider, img_size, img_aspect, img_quality],
                    outputs=[img_output, img_status],
                )

            # ----------------------------------------------------------
            # Video generation
            # ----------------------------------------------------------
            with gr.Tab("Video"):
                gr.Markdown("### 動画生成")
                gr.Markdown(
                    "> Veo による動画生成は完了まで数分かかる場合があります。"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        vid_prompt = gr.Textbox(
                            label="プロンプト",
                            placeholder="A drone shot flying over a coral reef...",
                            lines=3,
                        )
                        with gr.Row():
                            vid_provider = gr.Dropdown(
                                label="Provider",
                                choices=["veo"],
                                value="veo",
                            )
                            vid_duration = gr.Slider(
                                label="Duration (sec)",
                                minimum=2,
                                maximum=10,
                                value=5,
                                step=1,
                            )
                            vid_aspect = gr.Dropdown(
                                label="Aspect Ratio",
                                choices=["16:9", "9:16", "1:1"],
                                value="16:9",
                            )
                        vid_btn = gr.Button("動画生成", variant="primary")
                        vid_status = gr.Textbox(label="ステータス", interactive=False)

                    with gr.Column(scale=2):
                        vid_output = gr.Video(label="生成結果")

                vid_btn.click(
                    fn=generate_video,
                    inputs=[vid_prompt, vid_provider, vid_duration, vid_aspect],
                    outputs=[vid_output, vid_status],
                )

            # ----------------------------------------------------------
            # Music generation
            # ----------------------------------------------------------
            with gr.Tab("Music"):
                gr.Markdown("### 音楽生成")
                with gr.Row():
                    with gr.Column(scale=2):
                        mus_prompt = gr.Textbox(
                            label="プロンプト",
                            placeholder="Upbeat lo-fi hip hop beat with piano and rain sounds...",
                            lines=3,
                        )
                        with gr.Row():
                            mus_provider = gr.Dropdown(
                                label="Provider",
                                choices=["musicfx"],
                                value="musicfx",
                            )
                            mus_duration = gr.Slider(
                                label="Duration (sec)",
                                minimum=10,
                                maximum=60,
                                value=30,
                                step=5,
                            )
                        mus_btn = gr.Button("音楽生成", variant="primary")
                        mus_status = gr.Textbox(label="ステータス", interactive=False)

                    with gr.Column(scale=2):
                        mus_output = gr.Audio(label="生成結果")

                mus_btn.click(
                    fn=generate_music,
                    inputs=[mus_prompt, mus_provider, mus_duration],
                    outputs=[mus_output, mus_status],
                )

    return tab
