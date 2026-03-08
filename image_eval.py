"""
Image Evaluation Pipeline for Hallucinated Memories Project
============================================================
Compares reference images to AI-generated reconstructions using
perceptual and semantic similarity metrics (not pixel-level).

Metrics:
  1. CLIP Similarity   – semantic/conceptual similarity in shared embedding space
  2. LPIPS             – learned perceptual similarity (deep feature distance)
  3. SSIM              – structural similarity (luminance, contrast, structure)
  4. Object-Level Diff – detects objects in both images, categorizes hallucinations
                         as additions, omissions, or distortions

Usage:
  # Compare two images
  python image_eval.py --reference path/to/original.jpg --generated path/to/generated.jpg

  # Compare one reference to a folder of generated images
  python image_eval.py --reference path/to/original.jpg --generated path/to/generated_folder/

  # Include text description for text-image alignment scoring
  python image_eval.py --reference original.jpg --generated gen.jpg \
      --description "A red barn in a green field with two horses"

  # Export results to CSV
  python image_eval.py --reference original.jpg --generated gen_folder/ --output results.csv
"""

import argparse
import json
import csv
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ─── Metric Imports ──────────────────────────────────────────────────────────

# CLIP: semantic similarity in shared text-image embedding space
import clip

# LPIPS: learned perceptual image patch similarity
import lpips

# SSIM: structural similarity
from skimage.metrics import structural_similarity as ssim_metric
from skimage.transform import resize as skimage_resize

# Object detection via open-vocabulary detector
from transformers import pipeline as hf_pipeline


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class SimilarityScores:
    """Container for all similarity metrics between two images."""
    clip_image_similarity: float = 0.0       # [0, 1] higher = more similar
    clip_text_alignment_ref: float = 0.0     # how well text matches reference
    clip_text_alignment_gen: float = 0.0     # how well text matches generated
    lpips_distance: float = 0.0              # [0, 1+] lower = more similar
    ssim_score: float = 0.0                  # [-1, 1] higher = more similar
    reference_path: str = ""
    generated_path: str = ""
    description: str = ""


@dataclass
class ObjectDiff:
    """Categorized differences between detected objects in two images."""
    additions: list = field(default_factory=list)    # in generated but not reference
    omissions: list = field(default_factory=list)    # in reference but not generated
    shared: list = field(default_factory=list)       # present in both
    reference_objects: list = field(default_factory=list)
    generated_objects: list = field(default_factory=list)


@dataclass
class EvalResult:
    """Full evaluation result for one image pair."""
    scores: SimilarityScores = field(default_factory=SimilarityScores)
    object_diff: Optional[ObjectDiff] = None


# ─── Image Loading Utilities ─────────────────────────────────────────────────

def load_image_pil(path: str) -> Image.Image:
    """Load an image as PIL RGB."""
    return Image.open(path).convert("RGB")


def load_image_tensor(path: str, size: int = 256) -> torch.Tensor:
    """Load an image as a normalized tensor for LPIPS."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # LPIPS expects images in [-1, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = load_image_pil(path)
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def load_image_numpy(path: str, size: int = 256) -> np.ndarray:
    """Load an image as a numpy array for SSIM."""
    img = load_image_pil(path)
    img = img.resize((size, size))
    return np.array(img)


# ─── Metric Calculators ──────────────────────────────────────────────────────

class CLIPScorer:
    """Computes semantic similarity using CLIP embeddings."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    @torch.no_grad()
    def image_similarity(self, img_path_a: str, img_path_b: str) -> float:
        """Cosine similarity between two images in CLIP space."""
        img_a = self.preprocess(load_image_pil(img_path_a)).unsqueeze(0).to(self.device)
        img_b = self.preprocess(load_image_pil(img_path_b)).unsqueeze(0).to(self.device)

        feat_a = self.model.encode_image(img_a)
        feat_b = self.model.encode_image(img_b)

        # Normalize and compute cosine similarity
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)

        similarity = (feat_a @ feat_b.T).item()
        return float(similarity)

    @torch.no_grad()
    def text_image_alignment(self, text: str, img_path: str) -> float:
        """Cosine similarity between text and image in CLIP space."""
        img = self.preprocess(load_image_pil(img_path)).unsqueeze(0).to(self.device)
        tokens = clip.tokenize([text], truncate=True).to(self.device)

        img_feat = self.model.encode_image(img)
        txt_feat = self.model.encode_text(tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        similarity = (img_feat @ txt_feat.T).item()
        return float(similarity)


class LPIPSScorer:
    """Computes learned perceptual distance using LPIPS."""

    def __init__(self, net: str = "alex", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)

    @torch.no_grad()
    def distance(self, img_path_a: str, img_path_b: str) -> float:
        """Perceptual distance: lower = more similar."""
        img_a = load_image_tensor(img_path_a).to(self.device)
        img_b = load_image_tensor(img_path_b).to(self.device)
        return self.model(img_a, img_b).item()


class SSIMScorer:
    """Computes structural similarity index."""

    def __init__(self, size: int = 256):
        self.size = size

    def score(self, img_path_a: str, img_path_b: str) -> float:
        """SSIM score: higher = more similar. Range [-1, 1]."""
        img_a = load_image_numpy(img_path_a, self.size)
        img_b = load_image_numpy(img_path_b, self.size)

        return float(ssim_metric(
            img_a, img_b,
            channel_axis=2,        # color images, channel is last axis
            data_range=255,
            win_size=7,
        ))


class ObjectDetector:
    """
    Detects objects in images using an open-vocabulary object detection model
    and computes set differences to categorize hallucinations.

    Uses OWL-ViT (Open-World Localization with Vision Transformers) which can
    detect objects from text queries without fine-tuning.
    """

    # Default object categories relevant to scene reconstruction.
    # Deduplicated: one label per concept to avoid double-counting.
    DEFAULT_LABELS = [
        # Objects actually in the scene
        "couch", "armchair", "pillow",
        "coffee table", "side table", "stool", "ottoman",
        "rug",
        "plant",
        "window", "painting", "picture frame",
        "book", "cabinet", "shelf", "floor",
        # Plausible hallucinations (things participants might falsely recall)
        "lamp", "blanket", "television", "remote control", "vase",
        "candle", "clock", "mirror", "cat", "dog", "cup",
        "curtain", "speaker", "phone", "magazine",
    ]

    # Maps synonyms detected by the model to a single canonical label.
    # The detector sometimes uses different words for the same object;
    # this ensures "sofa" and "couch" aren't counted as two different things.
    SYNONYM_MAP = {
        "sofa": "couch",
        "cushion": "pillow",
        "carpet": "rug",
        "potted plant": "plant",
        "houseplant": "plant",
        "art": "painting",
        "pouf": "ottoman",
        "mug": "cup",
        "hardwood floor": "floor"
    }

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        labels: list = None,
        confidence_threshold: float = 0.25,
        generated_threshold: float = None,
        device: str = None,
        synonym_map: dict = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_id = 0 if self.device == "cuda" else -1
        self.detector = hf_pipeline(
            "zero-shot-object-detection",
            model=model_name,
            device=device_id,
        )
        self.labels = labels or self.DEFAULT_LABELS
        self.threshold = confidence_threshold
        # AI-generated images have softer features → lower confidence scores.
        # Use a separate (lower) threshold for generated images.
        self.generated_threshold = generated_threshold or (confidence_threshold * 0.6)
        self.synonym_map = synonym_map or self.SYNONYM_MAP

    def _normalize_label(self, label: str) -> str:
        """Map a detected label to its canonical form."""
        return self.synonym_map.get(label, label)

    def detect(self, img_path: str, threshold: float = None) -> list[dict]:
        """Detect objects in an image. Returns list of {label, score, box}."""
        image = load_image_pil(img_path)
        t = threshold or self.threshold
        results = self.detector(
            image,
            candidate_labels=self.labels,
            threshold=t,
        )
        return [
            {
                "label": self._normalize_label(r["label"]),
                "score": round(r["score"], 3),
                "box": r["box"],
            }
            for r in results
        ]

    def diff(self, ref_path: str, gen_path: str) -> ObjectDiff:
        """
        Compare detected objects between reference and generated images.
        Uses a higher threshold for reference (real photos) and a lower
        threshold for generated images (AI-generated, softer features).
        """
        ref_objects = self.detect(ref_path, threshold=self.threshold)
        gen_objects = self.detect(gen_path, threshold=self.generated_threshold)

        ref_labels = set(obj["label"] for obj in ref_objects)
        gen_labels = set(obj["label"] for obj in gen_objects)

        return ObjectDiff(
            additions=sorted(gen_labels - ref_labels),
            omissions=sorted(ref_labels - gen_labels),
            shared=sorted(ref_labels & gen_labels),
            reference_objects=[obj["label"] for obj in ref_objects],
            generated_objects=[obj["label"] for obj in gen_objects],
        )


# ─── Main Evaluation Pipeline ────────────────────────────────────────────────

class ImageEvaluator:
    """
    Full evaluation pipeline combining all metrics.

    Example:
        evaluator = ImageEvaluator()
        result = evaluator.evaluate(
            "reference.jpg",
            "generated.jpg",
            description="A red barn with two horses in a green field"
        )
        evaluator.print_result(result)
    """

    def __init__(
        self,
        use_clip: bool = True,
        use_lpips: bool = True,
        use_ssim: bool = True,
        use_objects: bool = True,
        object_labels: list = None,
        object_threshold: float = 0.25,
        generated_threshold: float = None,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ImageEvaluator] Using device: {self.device}")

        self.clip_scorer = CLIPScorer(device=self.device) if use_clip else None
        self.lpips_scorer = LPIPSScorer(device=self.device) if use_lpips else None
        self.ssim_scorer = SSIMScorer() if use_ssim else None
        self.object_detector = (
            ObjectDetector(
                labels=object_labels,
                confidence_threshold=object_threshold,
                generated_threshold=generated_threshold,
                device=self.device,
            )
            if use_objects
            else None
        )

    def evaluate(
        self,
        reference_path: str,
        generated_path: str,
        description: str = "",
    ) -> EvalResult:
        """Run all enabled metrics on one image pair."""
        scores = SimilarityScores(
            reference_path=reference_path,
            generated_path=generated_path,
            description=description,
        )

        # CLIP image-image similarity
        if self.clip_scorer:
            scores.clip_image_similarity = self.clip_scorer.image_similarity(
                reference_path, generated_path
            )
            # CLIP text-image alignment (if description provided)
            if description:
                scores.clip_text_alignment_ref = self.clip_scorer.text_image_alignment(
                    description, reference_path
                )
                scores.clip_text_alignment_gen = self.clip_scorer.text_image_alignment(
                    description, generated_path
                )

        # LPIPS perceptual distance
        if self.lpips_scorer:
            scores.lpips_distance = self.lpips_scorer.distance(
                reference_path, generated_path
            )

        # SSIM structural similarity
        if self.ssim_scorer:
            scores.ssim_score = self.ssim_scorer.score(
                reference_path, generated_path
            )

        # Object-level diff
        obj_diff = None
        if self.object_detector:
            obj_diff = self.object_detector.diff(reference_path, generated_path)

        return EvalResult(scores=scores, object_diff=obj_diff)

    def evaluate_batch(
        self,
        reference_path: str,
        generated_paths: list[str],
        descriptions: list[str] = None,
    ) -> list[EvalResult]:
        """Evaluate a reference image against multiple generated images."""
        if descriptions is None:
            descriptions = [""] * len(generated_paths)
        elif len(descriptions) == 1:
            descriptions = descriptions * len(generated_paths)

        results = []
        for i, (gen_path, desc) in enumerate(zip(generated_paths, descriptions)):
            print(f"  [{i + 1}/{len(generated_paths)}] Evaluating {Path(gen_path).name}...")
            results.append(self.evaluate(reference_path, gen_path, desc))
        return results

    @staticmethod
    def print_result(result: EvalResult):
        """Pretty-print evaluation results."""
        s = result.scores
        print("\n" + "=" * 60)
        print(f"  Reference : {Path(s.reference_path).name}")
        print(f"  Generated : {Path(s.generated_path).name}")
        if s.description:
            print(f"  Description: \"{s.description[:80]}...\"" if len(s.description) > 80 else f"  Description: \"{s.description}\"")
        print("-" * 60)

        print(f"  CLIP Image Similarity  : {s.clip_image_similarity:.4f}  (1.0 = identical)")
        if s.description:
            print(f"  CLIP Text↔Ref Align   : {s.clip_text_alignment_ref:.4f}")
            print(f"  CLIP Text↔Gen Align   : {s.clip_text_alignment_gen:.4f}")
            drift = s.clip_text_alignment_ref - s.clip_text_alignment_gen
            print(f"  Text Alignment Drift  : {drift:+.4f}  (+ = ref closer to text)")

        print(f"  LPIPS Distance         : {s.lpips_distance:.4f}  (0.0 = identical)")
        print(f"  SSIM Score             : {s.ssim_score:.4f}  (1.0 = identical)")

        if result.object_diff:
            od = result.object_diff
            print("-" * 60)
            print(f"  Objects in reference   : {od.reference_objects}")
            print(f"  Objects in generated   : {od.generated_objects}")
            print(f"  Shared (correct)       : {od.shared}")
            print(f"  Additions (AI hallu.)  : {od.additions}")
            print(f"  Omissions (forgotten)  : {od.omissions}")

        print("=" * 60)

    @staticmethod
    def results_to_csv(results: list[EvalResult], output_path: str):
        """Export evaluation results to CSV."""
        rows = []
        for r in results:
            row = asdict(r.scores)
            if r.object_diff:
                row["additions"] = "; ".join(r.object_diff.additions)
                row["omissions"] = "; ".join(r.object_diff.omissions)
                row["shared_objects"] = "; ".join(r.object_diff.shared)
                row["n_ref_objects"] = len(r.object_diff.reference_objects)
                row["n_gen_objects"] = len(r.object_diff.generated_objects)
            rows.append(row)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perceptual similarity between reference and generated images."
    )
    parser.add_argument(
        "--reference", "-r", required=True,
        help="Path to the reference (original) image."
    )
    parser.add_argument(
        "--generated", "-g", required=True,
        help="Path to a generated image or folder of generated images."
    )
    parser.add_argument(
        "--description", "-d", default="",
        help="Text description (participant recollection) for text-image alignment."
    )
    parser.add_argument(
        "--output", "-o", default="",
        help="Path to save results as CSV."
    )
    parser.add_argument(
        "--no-clip", action="store_true", help="Disable CLIP scoring."
    )
    parser.add_argument(
        "--no-lpips", action="store_true", help="Disable LPIPS scoring."
    )
    parser.add_argument(
        "--no-ssim", action="store_true", help="Disable SSIM scoring."
    )
    parser.add_argument(
        "--no-objects", action="store_true", help="Disable object detection."
    )
    parser.add_argument(
        "--object-labels", nargs="+", default=None,
        help="Custom object labels for detection (space-separated)."
    )
    parser.add_argument(
        "--object-threshold", type=float, default=0.25,
        help="Confidence threshold for reference image object detection (default: 0.25)."
    )
    parser.add_argument(
        "--generated-threshold", type=float, default=None,
        help="Confidence threshold for generated image detection (default: 60%% of --object-threshold). "
             "AI-generated images have softer features and need a lower threshold."
    )
    args = parser.parse_args()

    # Resolve generated image paths
    gen_path = Path(args.generated)
    if gen_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        generated_paths = sorted(
            str(p) for p in gen_path.iterdir() if p.suffix.lower() in extensions
        )
        if not generated_paths:
            print(f"No image files found in {gen_path}")
            return
        print(f"Found {len(generated_paths)} images in {gen_path}")
    else:
        generated_paths = [str(gen_path)]

    # Initialize evaluator
    evaluator = ImageEvaluator(
        use_clip=not args.no_clip,
        use_lpips=not args.no_lpips,
        use_ssim=not args.no_ssim,
        use_objects=not args.no_objects,
        object_labels=args.object_labels,
        object_threshold=args.object_threshold,
        generated_threshold=args.generated_threshold,
    )

    # Run evaluation
    results = evaluator.evaluate_batch(
        args.reference, generated_paths, [args.description]
    )

    # Print results
    for result in results:
        evaluator.print_result(result)

    # Export CSV if requested
    if args.output:
        evaluator.results_to_csv(results, args.output)


if __name__ == "__main__":
    main()