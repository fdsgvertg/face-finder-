#!/usr/bin/env python3
"""
Face Finder - Find all photos of a specific person from a folder of images.

Usage:
    python face_finder.py --reference path/to/person.jpg --folder path/to/photos/ --output matches/

Or interactive mode:
    python face_finder.py
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    print("Error: OpenCV is required. Install with: pip install opencv-python-headless")
    sys.exit(1)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}

# Default model paths (relative to this script or absolute)
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DET_MODEL = SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_REC_MODEL = SCRIPT_DIR / "models" / "face_recognition_sface_2021dec.onnx"

# Matching thresholds (OpenCV SFace recommended)
# Cosine similarity: higher is more similar. Threshold ~0.363
COSINE_THRESHOLD = 0.363
# L2 norm: lower is more similar. Threshold ~1.128
L2_THRESHOLD = 1.128


def load_models(det_model: Path, rec_model: Path, score_threshold: float = 0.6):
    """Load YuNet face detector and SFace recognizer."""
    if not det_model.exists():
        raise FileNotFoundError(f"Detection model not found: {det_model}")
    if not rec_model.exists():
        raise FileNotFoundError(f"Recognition model not found: {rec_model}")

    detector = cv2.FaceDetectorYN.create(
        str(det_model),
        "",
        (320, 320),
        score_threshold=score_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(str(rec_model), "")
    return detector, recognizer


def detect_faces(detector: cv2.FaceDetectorYN, image: np.ndarray) -> np.ndarray:
    """Detect faces in an image. Returns Nx15 array of detections."""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None:
        return np.array([])
    return faces


def get_face_embedding(
    recognizer: cv2.FaceRecognizerSF,
    image: np.ndarray,
    face: np.ndarray,
) -> Optional[np.ndarray]:
    """Align face and extract embedding feature."""
    try:
        # face is [x, y, w, h, ...]
        aligned = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned)
        return feature
    except Exception:
        return None


def compare_faces(
    recognizer: cv2.FaceRecognizerSF,
    feat1: np.ndarray,
    feat2: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Return similarity/distance score."""
    if metric == "cosine":
        return float(recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
    else:
        return float(recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_NORM_L2))


def is_match(score: float, metric: str = "cosine") -> bool:
    if metric == "cosine":
        return score >= COSINE_THRESHOLD
    return score <= L2_THRESHOLD


def extract_reference_embeddings(
    detector, recognizer, reference_path: Path
) -> List[np.ndarray]:
    """Load reference image and extract all face embeddings from it."""
    img = cv2.imread(str(reference_path))
    if img is None:
        raise ValueError(f"Could not read reference image: {reference_path}")

    faces = detect_faces(detector, img)
    if len(faces) == 0:
        raise ValueError(
            f"No face detected in reference image: {reference_path}\n"
            "Please use a clear photo of the person's face."
        )

    embeddings = []
    for face in faces:
        feat = get_face_embedding(recognizer, img, face)
        if feat is not None:
            embeddings.append(feat)

    if not embeddings:
        raise ValueError("Failed to extract face features from reference image.")

    print(f"  Found {len(embeddings)} face(s) in reference image.")
    return embeddings


def find_images(root: Path) -> List[Path]:
    """Recursively find all image files under root."""
    images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def process_folder(
    detector,
    recognizer,
    reference_feats: List[np.ndarray],
    folder: Path,
    output_dir: Optional[Path] = None,
    copy_matches: bool = True,
    metric: str = "cosine",
    min_confidence: float = COSINE_THRESHOLD,
) -> List[Tuple[Path, float]]:
    """
    Scan folder for matching faces.
    Returns list of (image_path, best_score) for matches.
    """
    images = find_images(folder)
    total = len(images)
    if total == 0:
        print(f"No images found in {folder}")
        return []

    print(f"Scanning {total} images in {folder} ...")
    matches = []

    for i, img_path in enumerate(images, 1):
        # Progress
        if i % 10 == 0 or i == total:
            print(f"  [{i}/{total}] processed...", end="\r", flush=True)

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = detect_faces(detector, img)
            if len(faces) == 0:
                continue

            best_score = -1.0 if metric == "cosine" else float("inf")
            matched = False

            for face in faces:
                feat = get_face_embedding(recognizer, img, face)
                if feat is None:
                    continue

                for ref_feat in reference_feats:
                    score = compare_faces(recognizer, ref_feat, feat, metric)
                    if metric == "cosine":
                        if score > best_score:
                            best_score = score
                        if score >= min_confidence:
                            matched = True
                    else:
                        if score < best_score:
                            best_score = score
                        if score <= L2_THRESHOLD:
                            matched = True

            if matched:
                matches.append((img_path, best_score))
                if output_dir and copy_matches:
                    # Preserve relative structure or just flat copy with unique name
                    rel = img_path.relative_to(folder)
                    dest = output_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest)

        except Exception as e:
            # Skip problematic images
            continue

    print(f"\n  Done. Found {len(matches)} matching photos.")
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Find all photos of a person from a folder using face recognition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_finder.py -r person.jpg -f ./photos -o ./matches
  python face_finder.py --reference me.png --folder /path/to/album --threshold 0.4
  python face_finder.py   # interactive mode
        """,
    )
    parser.add_argument(
        "-r", "--reference", type=str, help="Path to a clear photo of the person to find"
    )
    parser.add_argument(
        "-f", "--folder", type=str, help="Root folder containing photos to search (recursive)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output folder to copy matching photos into (optional)"
    )
    parser.add_argument(
        "--threshold", type=float, default=COSINE_THRESHOLD,
        help=f"Cosine similarity threshold (default: {COSINE_THRESHOLD}). Higher = stricter"
    )
    parser.add_argument(
        "--det-model", type=str, default=str(DEFAULT_DET_MODEL),
        help="Path to YuNet detection ONNX model"
    )
    parser.add_argument(
        "--rec-model", type=str, default=str(DEFAULT_REC_MODEL),
        help="Path to SFace recognition ONNX model"
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Only list matches, do not copy files"
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only print matching file paths (no scores or copy)"
    )

    args = parser.parse_args()

    # Interactive mode if arguments missing
    if not args.reference or not args.folder:
        print("=" * 60)
        print("  FACE FINDER - Find photos of a specific person")
        print("=" * 60)
        print()
        if not args.reference:
            args.reference = input("Enter path to reference photo of the person: ").strip().strip('"')
        if not args.folder:
            args.folder = input("Enter path to folder of photos to search: ").strip().strip('"')
        if not args.output:
            out = input("Enter output folder for matches (leave empty to only list): ").strip().strip('"')
            args.output = out if out else None

    reference_path = Path(args.reference).expanduser().resolve()
    folder_path = Path(args.folder).expanduser().resolve()

    if not reference_path.is_file():
        print(f"Error: Reference image not found: {reference_path}")
        sys.exit(1)
    if not folder_path.is_dir():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    output_dir = None
    if args.output and not args.no_copy:
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading face models...")
    try:
        detector, recognizer = load_models(
            Path(args.det_model), Path(args.rec_model)
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure the model files are present in the 'models' folder.")
        sys.exit(1)

    print(f"Extracting face features from reference: {reference_path.name}")
    try:
        ref_feats = extract_reference_embeddings(detector, recognizer, reference_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nSearching in: {folder_path}")
    matches = process_folder(
        detector,
        recognizer,
        ref_feats,
        folder_path,
        output_dir=output_dir,
        copy_matches=not args.no_copy and output_dir is not None,
        metric="cosine",
        min_confidence=args.threshold,
    )

    # Report results
    print("\n" + "=" * 60)
    if not matches:
        print("No matching photos found.")
        print("Tips: Try a clearer reference photo, lower --threshold (e.g. 0.30),")
        print("      or ensure the person appears with a visible face in the photos.")
    else:
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        print(f"Found {len(matches)} photos of this person:\n")
        for path, score in matches:
            if args.list_only:
                print(path)
            else:
                print(f"  [{score:.3f}]  {path}")

        if output_dir and not args.no_copy:
            print(f"\nMatching photos copied to: {output_dir}")

    print("=" * 60)


if __name__ == "__main__":
    main()
