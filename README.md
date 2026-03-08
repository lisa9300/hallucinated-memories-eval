# Hallucinated Memories — Image Evaluation Pipeline

Compares reference images to AI-generated reconstructions using perceptual and semantic similarity metrics. Built for the *Hallucinated Memories* project.

## Setup

### 1. Create a conda environment

```bash
conda create -n hallucination python=3.11 -y
conda activate hallucination
```

### 2. Install dependencies

```bash
python -m pip install torch torchvision numpy scikit-image lpips transformers Pillow
python -m pip install git+https://github.com/openai/CLIP.git
```

> **Note:** Always use `python -m pip` instead of bare `pip` to ensure packages install into the conda environment.

## Project Structure

```
Evaluation/
├── image_eval.py              # Main evaluation pipeline
├── quickstart.py              # Example usage as a Python library (optional)
├── requirements.txt           # Dependencies
├── README.md
├── reference/
│   └── scene_01.jpg           # Original image shown to participants
├── generated/
│   ├── example_01.jpg     # Example test AI-generated image 1
│   └── example_02.jpg     # Example test AI-generated image 2
└── results/
    └── scene_01_results.csv   # Evaluation output
```

## Usage

Make sure you're in the hallucination environment before running:

```bash
conda activate hallucination
```

### Compare a single generated image to the reference

```bash
python image_eval.py -r reference/scene_01.jpg -g generated/example_01.jpg
```

### Include a participant's text description

Adding `-d` enables CLIP text-image alignment scoring, which measures where distortion happens (human recall vs. AI generation):

```bash
python image_eval.py \
    -r reference/scene_01.jpg \
    -g generated/example_01.jpg \
    -d "A brown couch with colorful pillows in a living room with plants, a clear coffee table, paintings on the wall, and a pink armchair"
```

### Save results to CSV

```bash
python image_eval.py \
    -r reference/scene_01.jpg \
    -g generated/example_01.jpg \
    -d "participant's description here" \
    -o results/scene_01_results.csv
```

### Batch evaluate all generated images in a folder

```bash
python image_eval.py \
    -r reference/scene_01.jpg \
    -g generated/ \
    -o results/scene_01_results.csv
```

This finds all images in `generated/` and compares each one to the reference.

## Metrics

### CLIP Image Similarity (0 to 1, higher = more similar)

Measures semantic/conceptual similarity using CLIP embeddings. Good for "is this the same kind of scene?" but not sensitive to fine details. Two living rooms will score high even if the objects differ.

### CLIP Text Alignment (when `-d` is provided)

- **Text↔Ref Align**: How well the participant's description matches the original image.
- **Text↔Gen Align**: How well the AI-generated image matches the description.
- **Text Alignment Drift**: Difference between the two. Negative means the AI followed the description more closely than the description matched the original — i.e., the distortion happened at the human recall stage.

### LPIPS Distance (0 to 1+, lower = more similar)

Learned perceptual distance using deep features. More sensitive than CLIP to textures, colors, and spatial layout.

### SSIM Score (-1 to 1, higher = more similar)

Structural similarity at the pixel level. Will be low for any reference vs. AI-generated comparison since the images differ at the pixel level even when semantically similar.

### Object Detection Diff

Uses OWL-ViT to detect objects in both images and categorizes differences:

- **Shared**: Objects present in both images.
- **Additions (AI hallucination)**: Objects in the generated image but not the reference.
- **Omissions (forgotten)**: Objects in the reference but not the generated image.

## Tuning

### Object detection thresholds

The pipeline uses separate confidence thresholds for reference images (default 0.25) and generated images (default 60% of that = 0.15), since AI-generated images produce softer features.

```bash
# Override the reference threshold
python image_eval.py -r reference/scene_01.jpg -g generated/example_01.jpg --object-threshold 0.30

# Override the generated image threshold separately
python image_eval.py -r reference/scene_01.jpg -g generated/example_01.jpg --generated-threshold 0.18
```

### Disabling individual metrics

```bash
python image_eval.py -r reference/scene_01.jpg -g generated/example_01.jpg \
    --no-clip --no-lpips --no-ssim --no-objects
```

## Interpreting Results

For a typical evaluation, look at the metrics together:

| What you want to know | Primary metric | Supporting metric |
|---|---|---|
| Overall scene similarity | CLIP Image Similarity | LPIPS Distance |
| Where distortion happened | Text Alignment Drift | — |
| What objects were hallucinated | Additions list | — |
| What objects were forgotten | Omissions list | — |
| Human recall accuracy | Text↔Ref Align | — |
| AI generation fidelity | Text↔Gen Align | — |

CLIP alone won't differentiate well between participants since it operates at a coarse semantic level. The object diff and text alignment drift are more informative for our hallucination analysis.
