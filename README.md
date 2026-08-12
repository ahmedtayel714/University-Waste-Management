# University Waste Management & Agricultural Field Monitoring System

Automated detection of green vegetation/waste in agricultural and campus
field imagery, using HSV color-space masking as a domain-specific noise
filter ahead of YOLOv8 training, then reprojecting detections onto the
original unmasked RGB image for human-interpretable output.

## Core hypothesis

Standard object detectors trained on raw RGB agricultural imagery are
confused by soil texture and dry yellow straw, which visually resemble or
occlude green plant matter. This project tests whether HSV green-masking as
a **training-time preprocessing step** (not a runtime dependency — inference
still runs on raw RGB) improves detection precision/recall over an unmasked
baseline, trained identically otherwise.

## Repository layout

```
src/
  preprocessing/
    hsv_mask.py         # HSV green masking: hard / soft strategies, green-ratio scoring
    dataset_prep.py     # dataset discovery, class remap/merge, split, masked-variant generation
  training/
    train.py            # YOLOv8 training wrapper, baseline-vs-masked ablation runner
  inference/
    predict.py          # inference on original RGB + green-ratio false-positive filter
    combined_predict.py # runs vegetation + waste models together, merges annotations
  analysis/
    leaf_counter.py     # watershed leaf splitting/counting on the green mask
  evaluation/
    metrics.py          # results.csv parsing, loss/mAP curves, baseline-vs-masked comparison
    report.py           # combined original->mask->detection->leaf-count figure
notebooks/
  colab_pipeline.ipynb  # end-to-end Colab GPU notebook (the primary way to run this project)
```

## Dataset

Default target: [Crop and Weed Detection Data with Bounding Boxes](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes)
(Kaggle, ravirajsinh45) — ~1300 real sesame-field images (512×512) with
YOLO-format bounding boxes for crop and weed against natural soil
background. We merge both classes into a single `green_vegetation` class
since both are green plant matter being distinguished from soil.

**The exact on-disk layout and class-id convention (commonly `0=crop`,
`1=weed`) should be verified after download** — `dataset_prep.inspect_class_distribution()`
prints the class counts before any merging happens; the Colab notebook does
this automatically in step 3, don't skip reading its output.

Swapping in a different dataset only requires it to be in YOLO format
(image + matching `.txt` per image); `discover_pairs()` auto-matches by
filename stem regardless of folder layout.

## Running it

This project is designed to run on **Google Colab with a GPU runtime**
(T4 free tier is sufficient for `yolov8s`, 512px, batch 16):

1. Open `notebooks/colab_pipeline.ipynb` in Colab.
2. `Runtime > Change runtime type > T4 GPU`.
3. Either push this repo to GitHub and set `REPO_URL` in the setup cell, or
   upload the `src/` folder into the Colab file browser manually.
4. Run cells top to bottom. You'll be prompted to upload your Kaggle API
   token (`kaggle.json`) once.
5. All datasets, weights, and report figures are written under
   `/content/drive/MyDrive/university-waste-management/`, so they persist
   across Colab session resets — no manual export step needed.

To run locally instead (CPU or local GPU): `pip install -r requirements.txt`,
then call the same `src/` functions directly — the notebook cells are thin
wrappers around them.

## Preprocessing design — masking strategies

Two masking modes are implemented for ablation, not just one, because a hard
black background risks becoming a shortcut cue the network learns instead of
real vegetation texture:

- **hard** (`apply_mask_hard`): zero non-green pixels. Maximum noise removal,
  maximum domain shift from unmasked inference-time images.
- **soft** (`apply_mask_soft`, default): desaturate/darken background rather
  than zero it — suppresses the soil/straw color signal while preserving
  spatial context (field horizon, planting rows) and staying visually closer
  to the raw images the model sees at inference time.

Report both in your results section; the gap between them is itself
evidence for *why* the soft variant is the better design choice, not just
that masking helps at all.

## Post-processing: green-ratio filter

At inference time (`predict.py`), an optional `green_ratio_threshold` drops
any detected box whose interior isn't actually majority-green in HSV space —
a cheap, interpretable false-positive filter that's easy to justify
mathematically in a report (unlike a learned NMS variant).

## Evaluation

`src/evaluation/metrics.py` produces the core comparison artifact: a bar
chart of final-epoch precision / recall / mAP50 / mAP50-95 for baseline vs
masked runs, plus per-run loss and mAP-over-epochs curves. These are the
figures the competition writeup should lead with.

`src/evaluation/report.py` produces the per-sample story artifact:
`plot_pipeline_grid()` renders **Original → Green Mask → Masked (training
view) → Final Detection → Leaf Count** side by side for a handful of
images — the single figure that shows the whole pipeline working on one
sample, rather than scattered across separate plots.

## Leaf counting (per-plant, not just per-detection)

`src/analysis/leaf_counter.py` splits and counts individual leaves within
each detected vegetation box using a **distance-transform watershed** on
the green mask — no additional training, no new dataset, works today.

This is a classical-CV estimate, not ground truth, and it's honest about
one specific limitation: it only splits two leaves where their mask
silhouettes have a visible "neck" between them. Leaves that fully overlap
in the 2D image projection collapse into one count — there's no depth
information to disambiguate them from a single photo. If counts look
systematically off, `fg_ratio` (in `count_leaves`) trades off over- vs
under-splitting; there's no dataset-independent default that's right for
every leaf shape and density.

## Waste detection (trash / paper / soil anomalies) — a second model

The vegetation model is intentionally single-purpose: the HSV mask filters
*for* green, so it structurally cannot detect non-green waste (plastic,
paper, general rubbish). Rather than retrofit those categories into the
vegetation model — which would risk changing the already-validated
baseline-vs-masked ablation results — waste detection is a **separate
YOLO model**, trained independently on a litter-detection dataset
(default: [TACO — Trash Annotations in Context, YOLO format](https://www.kaggle.com/datasets/vencerlanz09/taco-dataset-yolo-format)),
and run alongside the vegetation model at inference time.

`src/inference/combined_predict.py` runs both models on the same image and
merges their detections into one annotated view — green boxes for
vegetation, red boxes for waste — without ever merging the models or
retraining either one on the other's data.

As with the crop/weed dataset, **TACO's exact category scheme must be
verified after download**, not assumed — the notebook's waste-detection
section prints the real class distribution and any discoverable class
names before you commit to a class-grouping map (e.g. collapsing ~60
fine-grained litter categories down to `plastic` / `paper` / `metal_glass`
/ `organic` / `other_rubbish`). `dataset_prep.split_dataset()`'s
`class_id_map` parameter handles the remap-and-drop in one pass.

## Requirements

See `requirements.txt`. Colab has most of these preinstalled; the notebook
installs the rest (`ultralytics`, `kaggle`, etc.) in its setup cell.

## Results

First completed ablation run (`yolov8s`, 512px, 100 epochs):

| Metric      | Baseline (unmasked) | HSV-Masked | Δ       |
|-------------|---------------------|------------|---------|
| Precision   | 0.822               | 0.885      | +0.063  |
| Recall      | 0.800               | 0.874      | +0.074  |
| mAP50       | 0.863               | 0.953      | +0.090  |
| mAP50-95    | 0.557               | 0.694      | +0.137  |

HSV green-masking as a training-time preprocessing step improved every
tracked metric, with the largest gain on mAP50-95 — the strictest,
localization-sensitive metric — supporting the core hypothesis that
filtering soil/straw noise before training sharpens the detector's spatial
precision, not just its coarse recall.
