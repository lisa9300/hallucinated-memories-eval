"""
Quickstart: Image Evaluation Pipeline
======================================

Demo/reference script

This script shows the three main ways to use the evaluator:
  1. Compare two images
  2. Compare one reference to many generated images
  3. Decompose where hallucination happens (human vs. AI)
"""

from image_eval import ImageEvaluator, CLIPScorer


# ── Example 1: Compare two images ────────────────────────────────────────────

evaluator = ImageEvaluator()

result = evaluator.evaluate(
    reference_path="images/original_scene.jpg",
    generated_path="images/generated_from_participant_01.jpg",
    description="A red barn in a green field with two horses and a fence",
)
evaluator.print_result(result)


# ── Example 2: Batch evaluation ──────────────────────────────────────────────

# Point to a folder of generated images — one result per image
results = evaluator.evaluate_batch(
    reference_path="images/original_scene.jpg",
    generated_paths=[
        "images/gen_participant_01.jpg",
        "images/gen_participant_02.jpg",
        "images/gen_aggregated_prompt.jpg",
        "images/gen_llm_synthesized.jpg",
    ],
    descriptions=[
        "A red barn with two horses and a white fence",       # participant 1
        "A farmhouse with one horse, surrounded by trees",    # participant 2
        "",   # aggregated prompt (no separate description)
        "",   # llm-synthesized (no separate description)
    ],
)

for r in results:
    evaluator.print_result(r)

# Export to CSV for further analysis
evaluator.results_to_csv(results, "evaluation_results.csv")


# ── Example 3: Decompose hallucination sources ───────────────────────────────
#
# Your project asks: where does distortion happen?
#   (a) Human recollection stage: participant description vs. original image
#   (b) AI generation stage: generated image vs. participant description
#   (c) End-to-end: generated image vs. original image
#
# CLIP text-image alignment lets you measure (a) and (b) separately.

clip_scorer = CLIPScorer()

original = "images/original_scene.jpg"
generated = "images/gen_participant_01.jpg"
description = "A red barn with two horses and a white fence"

# (a) How well does the participant's description match the ORIGINAL?
human_accuracy = clip_scorer.text_image_alignment(description, original)

# (b) How well does the generated image match the participant's DESCRIPTION?
ai_faithfulness = clip_scorer.text_image_alignment(description, generated)

# (c) How similar is the generated image to the original?
end_to_end = clip_scorer.image_similarity(original, generated)

print(f"\n{'='*60}")
print(f"  Hallucination Source Decomposition")
print(f"  {'─'*56}")
print(f"  (a) Human recall accuracy  (text ↔ original)  : {human_accuracy:.4f}")
print(f"  (b) AI generation fidelity (text ↔ generated) : {ai_faithfulness:.4f}")
print(f"  (c) End-to-end similarity  (original ↔ gen)   : {end_to_end:.4f}")
print(f"  {'─'*56}")
print(f"  Human distortion  = 1 - (a)                   : {1 - human_accuracy:.4f}")
print(f"  AI distortion     = (a) - (c)                 : {human_accuracy - end_to_end:.4f}")
print(f"{'='*60}")
