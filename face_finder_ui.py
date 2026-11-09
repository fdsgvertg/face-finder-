#!/usr/bin/env python3
"""
Face Finder - Simple Web UI
Find all photos of a specific person from a folder.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import gradio as gr

# -------------------------------------------------
# Model paths
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DET_MODEL = SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx"
REC_MODEL = SCRIPT_DIR / "models" / "face_recognition_sface_2021dec.onnx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Default threshold (cosine similarity)
DEFAULT_THRESHOLD = 0.363

# Global models (loaded once)
detector = None
recognizer = None


def load_models():
    global detector, recognizer
    if detector is not None:
        return

    if not DET_MODEL.exists() or not REC_MODEL.exists():
        raise FileNotFoundError(
            "Model files missing. Make sure the 'models' folder contains:\n"
            "  - face_detection_yunet_2023mar.onnx\n"
            "  - face_recognition_sface_2021dec.onnx"
        )

    detector = cv2.FaceDetectorYN.create(
        str(DET_MODEL), "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(REC_MODEL), "")


def detect_faces(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    return faces if faces is not None else np.array([])


def get_embedding(image: np.ndarray, face: np.ndarray) -> Optional[np.ndarray]:
    try:
        aligned = recognizer.alignCrop(image, face)
        return recognizer.feature(aligned)
    except Exception:
        return None


def compare(feat1: np.ndarray, feat2: np.ndarray) -> float:
    return float(recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))


def extract_reference_embeddings(ref_path: str) -> List[np.ndarray]:
    img = cv2.imread(ref_path)
    if img is None:
        raise ValueError("Could not read the reference image.")

    faces = detect_faces(img)
    if len(faces) == 0:
        raise ValueError(
            "No face detected in the reference photo.\n"
            "Please use a clear photo where the face is visible."
        )

    embeddings = []
    for face in faces:
        feat = get_embedding(img, face)
        if feat is not None:
            embeddings.append(feat)

    if not embeddings:
        raise ValueError("Failed to extract face features from the reference photo.")
    return embeddings


def find_images(root: Path) -> List[Path]:
    images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def search_faces(
    reference_img,
    folder_path: str,
    threshold: float,
    progress=gr.Progress(track_tqdm=False)
):
    """Main search function called by the UI."""
    load_models()

    if reference_img is None:
        return None, "Please upload a reference photo of the person.", []

    if not folder_path or not folder_path.strip():
        return None, "Please enter the path to the folder containing photos.", []

    folder = Path(folder_path.strip()).expanduser().resolve()
    if not folder.is_dir():
        return None, f"Folder not found:\n{folder}", []

    # Save uploaded reference temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        ref_path = tmp.name
        # reference_img is a numpy array (from Gradio Image)
        if isinstance(reference_img, np.ndarray):
            # Gradio gives RGB, OpenCV wants BGR
            cv2.imwrite(ref_path, cv2.cvtColor(reference_img, cv2.COLOR_RGB2BGR))
        else:
            return None, "Invalid reference image.", []

    try:
        progress(0.05, desc="Extracting face from reference photo...")
        ref_feats = extract_reference_embeddings(ref_path)

        progress(0.1, desc="Scanning folder for images...")
        images = find_images(folder)
        total = len(images)

        if total == 0:
            return None, f"No images found in:\n{folder}", []

        matches = []  # list of (path, score)
        gallery_items = []  # for Gradio Gallery

        for i, img_path in enumerate(images):
            progress(0.1 + 0.85 * (i / total), desc=f"Checking {i+1}/{total} ...")

            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                faces = detect_faces(img)
                if len(faces) == 0:
                    continue

                best_score = -1.0
                matched = False

                for face in faces:
                    feat = get_embedding(img, face)
                    if feat is None:
                        continue
                    for ref_feat in ref_feats:
                        score = compare(ref_feat, feat)
                        if score > best_score:
                            best_score = score
                        if score >= threshold:
                            matched = True

                if matched:
                    matches.append((str(img_path), best_score))
                    # Convert to RGB for Gradio gallery
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    gallery_items.append((rgb, f"{best_score:.3f}  |  {img_path.name}"))

            except Exception:
                continue

        # Sort by score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        gallery_items = [item for _, item in sorted(
            zip([m[1] for m in matches], gallery_items), reverse=True
        )]

        # Build result text
        if not matches:
            msg = (
                f"Scanned **{total}** images.\n\n"
                f"No matching photos found with threshold **{threshold:.3f}**.\n\n"
                "Tips:\n"
                "- Try a clearer / closer face photo as reference\n"
                "- Lower the threshold slider (e.g. 0.30)\n"
                "- Make sure the person appears with a visible face"
            )
            return None, msg, []
        else:
            lines = [f"**Found {len(matches)} matching photos** (out of {total} scanned)\n"]
            for path, score in matches[:50]:  # show top 50 in text
                lines.append(f"- `{score:.3f}`  {path}")
            if len(matches) > 50:
                lines.append(f"\n... and {len(matches)-50} more")
            msg = "\n".join(lines)
            return gallery_items, msg, [m[0] for m in matches]

    finally:
        # Clean up temp reference file
        try:
            os.unlink(ref_path)
        except Exception:
            pass


def copy_matches(match_paths: List[str], output_folder: str):
    """Copy matched photos to an output folder."""
    if not match_paths:
        return "No matches to copy."

    if not output_folder or not output_folder.strip():
        return "Please enter an output folder path."

    out_dir = Path(output_folder.strip()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in match_paths:
        try:
            src_path = Path(src)
            dest = out_dir / src_path.name
            # Avoid overwriting if same name exists
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                counter = 1
                while dest.exists():
                    dest = out_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(src_path, dest)
            copied += 1
        except Exception:
            continue

    return f"Successfully copied **{copied}** photos to:\n`{out_dir}`"


# -------------------------------------------------
# Build the Gradio UI
# -------------------------------------------------
with gr.Blocks(title="Face Finder") as demo:

    gr.Markdown(
        """
        # Face Finder
        <p class="subtitle">Upload a photo of a person → Select a folder → Find all their photos</p>
        """,
        elem_classes=["main-title"]
    )

    with gr.Row():
        with gr.Column(scale=1):
            reference = gr.Image(
                label="1. Reference Photo (the person you want to find)",
                type="numpy",
                height=280,
            )
            folder_input = gr.Textbox(
                label="2. Folder path to search (all subfolders included)",
                placeholder="e.g. /home/user/Pictures/Vacation or C:\\Users\\You\\Photos",
                lines=1,
            )
            threshold = gr.Slider(
                minimum=0.25,
                maximum=0.55,
                value=DEFAULT_THRESHOLD,
                step=0.01,
                label="Match Strictness (higher = fewer but more accurate results)",
                info="Recommended: 0.36 – 0.42",
            )
            search_btn = gr.Button("Search for this person", variant="primary", size="lg")

        with gr.Column(scale=1):
            status = gr.Markdown("Results will appear here after searching.")
            gallery = gr.Gallery(
                label="Matching Photos",
                columns=3,
                height=400,
                object_fit="contain",
                preview=True,
            )

    with gr.Accordion("Copy matching photos to a folder", open=False):
        with gr.Row():
            output_folder = gr.Textbox(
                label="Output folder path",
                placeholder="e.g. /home/user/Found_Photos",
                scale=3,
            )
            copy_btn = gr.Button("Copy Matches", variant="secondary", scale=1)
        copy_status = gr.Markdown("")

    # Hidden state to keep the list of matched paths
    match_paths_state = gr.State([])

    # Events
    search_btn.click(
        fn=search_faces,
        inputs=[reference, folder_input, threshold],
        outputs=[gallery, status, match_paths_state],
    )

    copy_btn.click(
        fn=copy_matches,
        inputs=[match_paths_state, output_folder],
        outputs=[copy_status],
    )

    gr.Markdown(
        """
        ---
        **Tips**
        - Use a clear, front-facing photo of the person as the reference
        - The search is recursive (looks inside all subfolders)
        - If you get too many wrong results → increase the threshold
        - If real photos are missing → decrease the threshold a little
        """
    )


if __name__ == "__main__":
    print("Starting Face Finder UI...")
    print("Models will be loaded on first search.")
    print("Open your browser at: http://127.0.0.1:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; }
        """,
    )
