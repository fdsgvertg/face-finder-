# Face Finder

A simple but powerful face recognition tool that finds **all photos of a specific person** inside a folder (and all its subfolders).

You give it:
1. A clear photo (or cropped face) of the person you want to find
2. A folder full of photos

It returns (and optionally copies) every photo that contains that person.

## Features

- Recursive search through all subfolders
- Uses modern OpenCV models (YuNet detector + SFace recognizer)
- Works offline after models are downloaded
- Adjustable similarity threshold
- Copies matching photos while preserving folder structure
- Interactive mode (just run the script)

## Requirements

- Python 3.8+
- OpenCV (`opencv-python` or `opencv-python-headless`)
- NumPy

Install OpenCV if needed:

```bash
pip install opencv-python-headless
```

## Models

The required ONNX models are already included in the `models/` folder:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`

## How to Use

### Web Interface (Recommended – Simple & Nice UI)

```bash
cd /home/workdir/artifacts
python face_finder_ui.py
```

Then open your browser and go to: **http://127.0.0.1:7860**

You will see a clean page where you can:
1. Upload the reference photo of the person
2. Type the folder path to search
3. Adjust the match strictness with a slider
4. Click **Search for this person**
5. View matching photos in a gallery
6. Optionally copy the matches to another folder

### Command-line Interactive mode

```bash
python face_finder.py
```

It will ask you for the paths.

### Command line (advanced)

```bash
python face_finder.py \
  --reference /path/to/person.jpg \
  --folder /path/to/all_photos \
  --output /path/to/matches
```

### Useful options

| Option | Description |
|--------|-------------|
| `-r`, `--reference` | Photo of the person to search for |
| `-f`, `--folder` | Root folder to search (recursive) |
| `-o`, `--output` | Folder where matching photos will be copied |
| `--threshold 0.36` | Similarity threshold (default 0.363). Lower = more results, higher = stricter |
| `--no-copy` | Only list matches, do not copy files |
| `--list-only` | Print only the file paths |

### Examples

```bash
# Find and copy matches
python face_finder.py -r me.jpg -f ~/Pictures/Vacation -o ~/Pictures/Me_in_Vacation

# Just list the matching files
python face_finder.py -r me.jpg -f ~/Pictures --list-only

# Stricter matching
python face_finder.py -r me.jpg -f ./photos -o ./found --threshold 0.42
```

## Tips for best results

1. **Reference photo** should be a clear, front-facing photo of the person (good lighting, face not too small).
2. If you get too many false positives → increase `--threshold` (try 0.40 – 0.45).
3. If you miss real matches → decrease `--threshold` (try 0.30 – 0.35).
4. Works best when the person's face is reasonably visible in the photos.

## How it works

1. Detects faces in the reference image and extracts a 128-dimensional face embedding (SFace).
2. Scans every image in the target folder.
3. Detects faces and extracts embeddings.
4. Compares embeddings using cosine similarity.
5. Keeps photos whose best face match exceeds the threshold.

---

Made for easy personal photo organization.
