"""ShadowFix: Gradio interface for shadow removal.

Wraps the project's inference.py with an Automatic mode (whole image goes
through the RGB residual U-Net) and a Guided mode (the user paints the shadow
and the restoration is blended into that region only).

Usage:
    python app/shadowfix_app.py --project-dir /path/to/shadow_removal_project \
        --checkpoint /path/to/best.pth [--share]
"""
import argparse
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import gradio as gr


parser = argparse.ArgumentParser(description="ShadowFix Gradio app")
parser.add_argument(
    "--project-dir",
    default=os.environ.get("SHADOWFIX_PROJECT_DIR", "."),
    help="Folder that contains inference.py"
)
parser.add_argument(
    "--checkpoint",
    default=os.environ.get("SHADOWFIX_CHECKPOINT", "checkpoints/best.pth"),
    help="Trained RGB U-Net checkpoint (.pth)"
)
parser.add_argument(
    "--output-dir",
    default=os.environ.get("SHADOWFIX_OUTPUT_DIR", "ui_outputs"),
    help="Where restored images are saved"
)
parser.add_argument(
    "--share",
    action="store_true",
    help="Create a public Gradio link (useful on Colab)"
)
args = parser.parse_args()

PROJECT_DIR = Path(args.project_dir).resolve()
CHECKPOINT = Path(args.checkpoint).resolve()
OUTPUT_DIR = Path(args.output_dir).resolve()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if not (PROJECT_DIR / "inference.py").exists():
    sys.exit(f"inference.py not found in {PROJECT_DIR}. Pass --project-dir.")

if not CHECKPOINT.exists():
    sys.exit(f"Checkpoint not found: {CHECKPOINT}. Pass --checkpoint.")


def pil_rgb(x):
    if x is None:
        return None

    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, np.ndarray):

        if x.ndim == 2:
            return Image.fromarray(
                x.astype(np.uint8)
            ).convert("RGB")

        if x.shape[-1] == 4:
            x = x[:, :, :3]

        return Image.fromarray(
            x.astype(np.uint8)
        ).convert("RGB")

    return Image.open(x).convert("RGB")


def model_restore_rgb(image):
    image = pil_rgb(image)

    if image is None:
        raise gr.Error(
            "Please upload an image."
        )

    work_dir = Path(
        tempfile.mkdtemp(
            prefix="shadowfix_rgb_"
        )
    )

    input_path = (
        work_dir / "input.png"
    )

    output_dir = (
        work_dir / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        str(input_path)
    )

    command = [
        sys.executable,
        "inference.py",

        "--checkpoint",
        str(CHECKPOINT),

        "--input",
        str(input_path),

        "--output-dir",
        str(output_dir),
    ]

    print("\n========================================")
    print("Running RGB-only ShadowFix model...")
    print("========================================")

    result = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)

        raise gr.Error(
            "Model inference failed.\n\n"
            + result.stderr[-4000:]
        )

    candidates = sorted(
        list(output_dir.glob("*.png"))
        + list(output_dir.glob("*.jpg"))
        + list(output_dir.glob("*.jpeg")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not candidates:
        raise gr.Error(
            "The model finished but no output image was created."
        )

    output_path = candidates[0]

    print(
        "Model output:",
        output_path
    )

    return Image.open(
        output_path
    ).convert("RGB")


def automatic_shadow_mask(image):

    image = pil_rgb(image)

    rgb = np.array(
        image
    )

    # RGB -> LAB
    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    L = (
        lab[:, :, 0]
        .astype(np.float32)
    )

    local_mean = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=35,
        sigmaY=35
    )
    darkness = (
        local_mean - L
    )

    mask = (
        darkness > 10
    ).astype(np.uint8) * 255



    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        small_kernel
    )



    large_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (25, 25)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        large_kernel
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    clean_mask = np.zeros_like(
        mask
    )

    min_area = 1500

    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            clean_mask[
                labels == i
            ] = 255

    return Image.fromarray(
        clean_mask,
        mode="L"
    )

def make_guided_mask(editor_data):

    if editor_data is None:
        raise gr.Error(
            "Please upload an image."
        )

    if not isinstance(
        editor_data,
        dict
    ):
        raise gr.Error(
            "Please use the Guided Mode image editor."
        )

    background = (
        editor_data.get("background")
    )

    layers = (
        editor_data.get("layers", [])
    )

    if background is None:
        raise gr.Error(
            "Please upload an image."
        )

    original = pil_rgb(
        background
    )

    width, height = (
        original.size
    )

    mask = None
    for layer in reversed(
        layers
    ):

        if layer is None:
            continue

        if isinstance(
            layer,
            Image.Image
        ):

            arr = np.array(
                layer
            )

        elif isinstance(
            layer,
            np.ndarray
        ):

            arr = layer

        else:
            continue
        if (
            arr.ndim == 3
            and arr.shape[2] == 4
        ):

            alpha = (
                arr[:, :, 3]
            )

            if np.max(alpha) > 0:

                mask = (
                    alpha.astype(
                        np.uint8
                    )
                )

                break
        elif arr.ndim == 3:

            rgb = (
                arr[:, :, :3]
            )

            gray = cv2.cvtColor(
                rgb.astype(
                    np.uint8
                ),
                cv2.COLOR_RGB2GRAY
            )

            if np.max(gray) > 0:

                mask = gray

                break

    if (
        mask is None
        or np.max(mask) == 0
    ):

        raise gr.Error(
            "No shadow was painted.\n\n"
            "Use the brush to paint over the shadow."
        )


    mask = cv2.resize(
        mask,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_LINEAR
    )


    mask = np.where(
        mask > 20,
        255,
        0
    ).astype(
        np.uint8
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.GaussianBlur(
        mask,
        (9, 9),
        0
    )

    return (
        original,
        mask
    )


def guided_blend(
    original,
    restored,
    strength,
    mask
):

    original_np = np.asarray(
        original.convert("RGB")
    ).astype(
        np.float32
    )

    restored_np = np.asarray(
        restored.convert("RGB")
    ).astype(
        np.float32
    )

    if (
        original_np.shape[:2]
        != restored_np.shape[:2]
    ):

        restored_np = cv2.resize(
            restored_np,
            (
                original_np.shape[1],
                original_np.shape[0]
            ),
            interpolation=cv2.INTER_CUBIC
        )

    mask_float = (
        np.asarray(mask)
        .astype(np.float32)
        / 255.0
    )
    mask_float = cv2.GaussianBlur(
        mask_float,
        (15, 15),
        0
    )

    amount = np.clip(
        float(strength),
        0,
        100
    ) / 100.0

    alpha = (
        mask_float[..., None]
        * amount
    )

    output = (
        original_np * (1.0 - alpha)
        +
        restored_np * alpha
    )

    output = np.clip(
        output,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        output,
        mode="RGB"
    )

def automatic_analysis(
    image,
    mask
):
    image_np = np.asarray(
        image.convert("RGB")
    )

    mask_np = np.asarray(
        mask.convert("L")
    )

    coverage = (
        np.count_nonzero(
            mask_np > 30
        )
        /
        mask_np.size
        * 100
    )

    if coverage < 5:
        severity = "Low"

    elif coverage < 15:
        severity = "Moderate"

    elif coverage < 35:
        severity = "High"

    else:
        severity = "Very High"

    return f"""
## AUTOMATIC ANALYSIS

| Metric | Result |
|---|---|
| Estimated Shadow Coverage | **{coverage:.1f}%** |
| Estimated Severity | **{severity}** |
| Model | **RGB Residual U-Net** |
| Model Input | **RGB image** |
| Shadow Mask Input | **Not required** |

### Automatic Processing

The RGB U-Net automatically predicts the shadow-free image.

The estimated mask shown beside the result is used for **analysis and visualization**; it is not required by the trained model.
"""

def automatic_restore(image):

    original = pil_rgb(
        image
    )

    if original is None:
        raise gr.Error(
            "Please upload an image."
        )
    restored = model_restore_rgb(
        original
    )
    estimated_mask = (
        automatic_shadow_mask(
            original
        )
    )

    analysis = automatic_analysis(
        original,
        estimated_mask
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = int(
        time.time()
    )

    output_path = (
        OUTPUT_DIR
        /
        f"shadowfix_automatic_{timestamp}.png"
    )

    mask_path = (
        OUTPUT_DIR
        /
        f"shadowfix_auto_mask_{timestamp}.png"
    )

    restored.save(
        str(output_path)
    )

    estimated_mask.save(
        str(mask_path)
    )

    print(
        "Automatic result saved:",
        output_path
    )

    return (
        original,
        restored,
        (original, restored),
        estimated_mask,
        analysis,
        str(output_path)
    )

def guided_restore(
    editor_data,
    mode,
    strength
):

    original, mask = (
        make_guided_mask(
            editor_data
        )
    )

    if mode == "Natural":
        strength = 55

    elif mode == "Balanced":
        strength = 80

    elif mode == "Strong":
        strength = 100

    restored = model_restore_rgb(
        original
    )

    final = guided_blend(
        original,
        restored,
        strength,
        mask
    )
    mask_np = np.asarray(
        mask
    )

    coverage = (
        np.count_nonzero(
            mask_np > 30
        )
        /
        mask_np.size
        * 100
    )

    if coverage < 10:
        severity = "Low"

    elif coverage < 25:
        severity = "Moderate"

    elif coverage < 45:
        severity = "High"

    else:
        severity = "Very High"

    analysis = f"""
## 🎯 GUIDED  ANALYSIS

| Metric | Result |
|---|---|
| Guided Shadow Coverage | **{coverage:.1f}%** |
| Shadow Severity | **{severity}** |
| Restoration Mode | **{mode}** |
| Strength | **{strength:.0f}%** |
| Model | **RGB Residual U-Net** |
| Guidance | **User-Painted Shadow Region** |

### Recommendation

**{mode}** mode is being used.

- Increase strength if shadow remains.
- Reduce strength if the restored region becomes too strong.
- Refine the painted mask if the wrong region is affected.
"""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = int(
        time.time()
    )

    final_path = (
        OUTPUT_DIR
        /
        f"shadowfix_guided_{timestamp}.png"
    )

    mask_path = (
        OUTPUT_DIR
        /
        f"shadowfix_guided_mask_{timestamp}.png"
    )

    final.save(
        str(final_path)
    )

    Image.fromarray(
        mask.astype(np.uint8)
    ).save(
        str(mask_path)
    )

    print(
        "Guided result saved:",
        final_path
    )

    return (
        original,
        final,
        (original, final),
        Image.fromarray(
            mask
        ),
        analysis,
        str(final_path)
    )
def update_guided_strength(
    mode,
    current
):

    if mode == "Natural":
        return 55

    if mode == "Balanced":
        return 80

    if mode == "Strong":
        return 100
    return current

def assistant(question):

    if not question:
        return (
            "Please ask a question about ShadowFix."
        )

    q = question.lower()

    if (
        "remaining" in q
        or "left" in q
        or "visible" in q
    ):
        return """
### Restoration Assistant

A remaining shadow can happen when the shadow is outside the model's learned distribution or when the selected guided region does not cover the complete shadow.

For Guided Mode, refine the painted region and try **Strong** mode.
"""

    if "automatic" in q:
        return """
### Restoration Assistant

**Automatic Mode** sends the uploaded image directly to the trained RGB Residual U-Net.

No user-painted mask is required.
"""

    if "mask" in q:
        return """
### Restoration Assistant

In Guided Mode, the mask identifies the region where you want the restoration to be emphasized.

The current trained model is **RGB-only**, so the mask is not fed into the network.
"""

    if "natural" in q:
        return """
### Restoration Assistant

**Natural — 55%**

A conservative blend that keeps more of the original image.
"""

    if "balanced" in q:
        return """
### ✨ Restoration Assistant

**Balanced — 80%**

Recommended starting point for Guided Mode.
"""

    if "strong" in q:
        return """
### Restoration Assistant

**Strong — 100%**

Uses the maximum guided blending of the model's restoration inside the selected shadow region.
"""

    if (
        "model" in q
        or "unet" in q
    ):
        return """
### Restoration Assistant

ShadowFix uses an **RGB Residual U-Net**.

The network takes the shadowed RGB image and predicts a restoration. Guided Mode then lets the user control where that restoration is emphasized.
"""

    if "slider" in q:
        return """
### Restoration Assistant

The Guided Mode slider controls how strongly the neural-network restoration is blended into the user-selected shadow region.

Try **80%** first.
"""

    return """
### Restoration Assistant

I can explain:

- Automatic Mode
- Guided Mode
- The shadow mask
- Natural / Balanced / Strong
- The restoration slider
- How the RGB Residual U-Net works
"""

with gr.Blocks(
    title="ShadowFix"
) as app:

    gr.Markdown(
        """
# ✨ ShadowFix
### Intelligent Shadow Removal & Image Restoration

**Automatic detection • Guided restoration • Before/After comparison**
"""
    )

    gr.Markdown(
        """
### Model
**RGB Residual U-Net**

The final interface uses the best RGB-only checkpoint from the capstone training pipeline.
"""
    )

    with gr.Tabs():

        with gr.Tab("🤖 Automatic Mode"):

            gr.Markdown(
                """
## 🤖 Automatic Shadow Removal

Upload your image and let ShadowFix automatically restore it.

**No painting or manual mask is required.**
"""
            )

            auto_input = gr.Image(
                type="pil",
                label="Upload Shadowed Image",
                height=450
            )

            auto_button = gr.Button(
                "AUTOMATICALLY REMOVE SHADOW",
                variant="primary",
                size="lg"
            )

            with gr.Row():

                auto_original = gr.Image(
                    label="ORIGINAL",
                    type="pil"
                )

                auto_restored = gr.Image(
                    label="RESTORED",
                    type="pil"
                )

            auto_comparison = gr.ImageSlider(
                label="🔄 BEFORE / AFTER",
                type="pil"
            )

            with gr.Row():

                auto_mask = gr.Image(
                    label="ESTIMATED SHADOW MAP",
                    type="pil"
                )

                auto_analysis = gr.Markdown()

            auto_download = gr.DownloadButton(
                label="⬇Download Automatic Result"
            )

            auto_button.click(
                automatic_restore,
                inputs=[
                    auto_input
                ],
                outputs=[
                    auto_original,
                    auto_restored,
                    auto_comparison,
                    auto_mask,
                    auto_analysis,
                    auto_download
                ]
            )
        with gr.Tab("Guided Mode"):

            gr.Markdown(
                """
## 🎯 Guided Shadow Removal

**1. Upload your image**
**2. Paint over the shadow**
**3. Use the eraser to fix mistakes**
**4. Select Natural / Balanced / Strong / Custom**
**5. Click RESTORE SHADOW**
"""
            )

            editor = gr.ImageEditor(
                label="Paint Over the Shadow",
                type="pil",
                image_mode="RGBA",
                height=500
            )

            gr.Markdown(
                """
 **Brush:** paint over the complete shadow
 **Eraser:** remove incorrect areas
 **Tip:** cover the whole shadow, including soft edges.
"""
            )

            guided_mode = gr.Radio(
                [
                    "Natural",
                    "Balanced",
                    "Strong",
                    "Custom"
                ],
                value="Balanced",
                label="Restoration Mode"
            )

            guided_strength = gr.Slider(
                minimum=0,
                maximum=100,
                value=80,
                step=5,
                label="Shadow Removal Strength"
            )

            guided_mode.change(
                update_guided_strength,
                inputs=[
                    guided_mode,
                    guided_strength
                ],
                outputs=[
                    guided_strength
                ]
            )

            guided_button = gr.Button(
                "✨ RESTORE SHADOW",
                variant="primary",
                size="lg"
            )

            with gr.Row():

                guided_original = gr.Image(
                    label="ORIGINAL",
                    type="pil"
                )

                guided_restored = gr.Image(
                    label="RESTORED",
                    type="pil"
                )

            guided_comparison = gr.ImageSlider(
                label="BEFORE / AFTER",
                type="pil"
            )

            with gr.Row():

                guided_mask_output = gr.Image(
                   label="GUIDED SHADOW MASK",
                   type="pil"
                )

                guided_analysis = gr.Markdown()

            guided_download = gr.DownloadButton(
                label="Download Guided Result"
            )

            guided_button.click(
                guided_restore,
                inputs=[
                    editor,
                    guided_mode,
                    guided_strength
                ],
                outputs=[
                    guided_original,
                    guided_restored,
                    guided_comparison,
                    guided_mask_output,
                    guided_analysis,
                    guided_download
                ]
            )

        with gr.Tab("Restoration Assistant"):

            gr.Markdown(
                """
# Restoration Assistant

Ask questions about the restoration process, model, mask, or settings.
"""
            )

            question = gr.Textbox(
                label="Ask a question",
                placeholder="How does Automatic Mode work?",
                lines=3
            )

            ask_button = gr.Button(
                "Ask Assistant",
                variant="primary"
            )

            answer = gr.Markdown()

            ask_button.click(
                assistant,
                inputs=[
                    question
                ],
                outputs=[
                    answer
                ]
            )

            gr.Examples(
                examples=[
                    [
                        "How does Automatic Mode work?"
                    ],
                    [
                        "How does Guided Mode work?"
                    ],
                    [
                        "Why is some shadow remaining?"
                    ],
                    [
                        "What does Strong mode do?"
                    ],
                    [
                        "How does the U-Net work?"
                    ]
                ],
                inputs=[
                    question
                ]
            )

    gr.Markdown(
        """
---


"""
    )

print("Starting ShadowFix ...")
print("Checkpoint:", CHECKPOINT)

app.launch(
    share=args.share,
    show_error=True,
    allowed_paths=[
        str(OUTPUT_DIR)
    ]
)
