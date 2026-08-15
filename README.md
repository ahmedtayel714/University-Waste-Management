# University Waste Management & Agricultural Field Monitoring System

Two separate tracks live in this repo. They share tooling (training
wrapper, metrics/report plotting) but not data, preprocessing, or results —
each is documented in its own section below.

- **Track A — Vegetation ablation** (complete, validated): does HSV
  green-masking as a training-time preprocessing step improve YOLO
  detection of green vegetation against soil/straw noise? Yes — see
  Results. Untouched by everything below.
- **Track B — Leaf detection for robotic collection** (new, in progress):
  detect every tree leaf (dry or fresh) in mixed campus waste, from a
  synthetically composited dataset, per a robotic-collection system spec.
  Single class `leaf`, plain-RGB training — green-masking doesn't apply
  since dry leaves aren't green.

---

# Track A — Vegetation Ablation

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

## Leaf counting within a vegetation blob (Track A only)

Track A's model detects one box per whole plant, not per leaf, so
`src/analysis/leaf_counter.py` splits and counts individual leaves within
each detected vegetation box using a **distance-transform watershed** on
the green mask — no additional training, no new dataset, works today. (This
is unrelated to Track B, whose leaf *detector* already produces one box per
leaf directly — no watershed post-processing needed there.)

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

---

# Track B — Leaf Detection for Robotic Collection

Detect every tree leaf (dry or fresh) inside mixed university outdoor
waste, count it, and expose detections through a structured JSON interface
a future robotic system can consume — following the full system
specification's own staged priority (dataset → detection → inference →
counting/tracking → *then* calibration/robot layers, never the robot
first). The physical robot does not exist; the software is built so it can
be added later without redesigning the detection pipeline.

## Why this can't reuse Track A's preprocessing

Track A's core technique — HSV masking to isolate green pixels — is
useless here: dry fallen leaves are brown/yellow, not green, so a green
filter would delete the actual detection target. Track B trains on
plain RGB, no color masking, single class `leaf`.

## Dataset strategy: no usable public dataset exists — composite one

Search turned up nothing usable for "fallen leaves mixed with campus
waste" or even a plain "empty ground" set (everything found was either
stock-photo licensed or the wrong domain — live crop leaves, disease-spot
leaves, pavement-distress research data). So the training set is
**synthetically composited from four real sources**, each solving a piece
neither of the others can:

| Source | Role | Where it comes from |
|---|---|---|
| Real leaf cutouts | The labeled detection target | Small Roboflow leaf-**segmentation** projects (need pixel masks, not just boxes, for clean silhouettes — see `src/synthetic/cutout_extractor.py`) |
| Real waste cutouts | Unlabeled visual clutter only — spec explicitly says not to add plastic/paper/etc. as classes | Reused from Track B's own TACO download (bbox crops, feathered edges) |
| Real backgrounds | Ground/soil/pavement scenes | **Harvested from datasets already on disk** — soil regions of Track A's crop/weed dataset (inverted green mask) + non-litter regions of TACO images. Real photos of real outdoor ground, zero new downloads. |
| Controlled synthetic generation | Combines the above with randomized geometry | `src/synthetic/compositor.py` — since *we* control the paste, YOLO boxes come out automatically; zero manual annotation |

This pass uses public sources only (no phone photos) by explicit choice —
the pipeline is dataset-agnostic by design (`discover_pairs`-style
stem-matching, mask-or-bbox cutout extraction, generic background
harvesting), so real campus photos can be dropped into the same folder
structure later with no code changes, matching the spec's own roadmap
(public data → synthetic → initial model → field validation → fine-tune).

`compositor.py`'s difficulty tiers (`easy`/`medium`/`hard` in
`DIFFICULTY_PRESETS`) control leaf scale/rotation range, overlap
probability, edge-cropping (partial occlusion), blur, and shadow — directly
covering the spec's Section 4 variation list (size, orientation, rotation,
overlap, occlusion, lighting/shadow, blur, background, density).

## Repository additions

```
src/
  synthetic/
    cutout_extractor.py      # YOLO-seg polygon -> RGBA cutout; bbox -> feathered RGBA; GrabCut fallback for unmasked sources
    background_harvester.py  # harvest object-free ground patches from datasets we already have
    compositor.py             # paste engine: randomized geometry, auto-bbox, shadow, difficulty tiers
    generate_dataset.py       # orchestrates compositor at scale -> full YOLO train/val/test tree
    pipeline.py                # single build_leaf_dataset() call wrapping all of the above
  inference/
    track.py                  # Ultralytics native tracking (ByteTrack) + the spec's exact JSON schemas + EMA smoothing
    error_logger.py            # low-confidence detection snapshots + JSONL index, for active learning
  evaluation/
    field_validation.py        # real-photo validation harness (detection stats, or real mAP if labeled)
```

Training and evaluation reuse Track A's `src/training/train.py` and
`src/evaluation/metrics.py` unchanged — the wrapper only cares about a
`data.yaml` path and hyperparameters, not what's in the dataset.

## Interface contract (built now, consumed by nothing yet)

`src/inference/track.py` formats every detection/track to match the
spec's own JSON shapes exactly (Section 3's per-detection object, Section
7's tracked-leaf object with persistent `id`, Section 20's `/detections`
response envelope). No API server or robot code exists yet — the point is
that the *shape* is fixed now, so a future FastAPI/Flask layer or a real
robot controller is a thin adapter over this, not a rewrite of the
detection code.

Tracking itself uses Ultralytics' built-in ByteTrack (`model.track()`)
rather than a custom tracker — the spec only needs stable per-leaf IDs
across frames, which the built-in tracker already provides.

## Deferred (by explicit scope decision, not oversight)

Camera calibration, pixel→world coordinate transforms, target selection,
grasp-point estimation, the robot command API, the robot state machine,
the safety layer, and the dashboard — spec Sections 8-14 and 16-19. These
are real, buildable pieces, intentionally left until the leaf detector's
accuracy is validated on this dataset; building a calibration module
against a detector that doesn't exist yet would be scaffolding for its own
sake. Instance segmentation (spec Section 15) is similarly deferred —
boxes are enough to prove detection works before upgrading to masks.

---

# Roadmap Improvements (Part 3)

A set of targeted improvements on top of the trained `leaf` model,
implemented from an internal improvements roadmap. **None of these change
Track A's baseline/masked results or the original `leaf` run** — every
new experiment gets a new name (`leaf_v2`) rather than overwriting an
already-reported result, and every new function parameter defaults to the
old behavior (off) unless explicitly opted into.

## Augmentation: mixup + random perspective

`TrainConfig` already had mosaic and HSV jitter; `mixup` and
`perspective` were the two missing pieces, added as opt-in fields
(default `0.0`, matching Ultralytics' pre-existing off-state, so any
existing `TrainConfig(...)` call reproduces identically unless it sets
these explicitly). Applied to a new `leaf_v2` run trained on the *same*
synthetic dataset as `leaf` — augmentation isolated as the only variable,
same ablation principle used for baseline-vs-masked in Track A.

## Test-time augmentation (TTA)

`augment: bool = False` threaded through every inference entry point
(`predict.py`, `report.py`, `track.py`, `combined_predict.py`).
Ultralytics runs multi-scale + flip inference and averages the result —
catches small/marginal detections at ~2-3x slower inference. Meant for a
final careful evaluation pass, not live video (the cost compounds per
frame there).

## Temporal smoothing — EMA, not Kalman

`TrackSmoother` in `track.py` applies an exponential moving average to
each tracked leaf's center coordinates, keyed by track id. This was a
deliberate simplification versus a full Kalman filter: a Kalman filter
models velocity/acceleration and needs process-noise tuning to get right;
EMA needs one parameter (`alpha`) and no motion model, and achieves the
same practical goal stated in the roadmap — smooth, non-jittery
coordinates for a downstream robot controller — with a fraction of the
code. Revisit with a real Kalman filter only if velocity/acceleration
estimates become actually necessary, not just smoothed position.
`track_video(..., smooth=True)` opts in; default is `False` so an
already-captured tracking sample stays reproducible.

## Unified Track B pipeline

`src/synthetic/pipeline.py`'s `build_leaf_dataset()` wraps cutout
extraction (leaf + waste) → background harvesting → synthetic generation
behind one call, for rebuilding the dataset from scratch without wiring
six notebook cells together by hand each time. The individual modules
stay independently usable — this is a convenience layer, not a
replacement API.

## Error logging for active learning

`LowConfidenceLogger` (`src/inference/error_logger.py`) saves an
annotated snapshot plus a JSONL index entry for every inference where at
least one detection falls under a confidence threshold (default `0.5`).
Run it over a test/validation split, or in production, and the output is
a ready-made relabeling queue — the exact "blacklist of challenging
scenarios" the roadmap asked for.

## Real-world field validation harness

`src/evaluation/field_validation.py`'s `run_field_validation()` is built
and tested, but **cannot run meaningfully yet — there are no real
validation photos**. Point it at a folder with an `images/` subfolder and
it reports detection-rate and confidence statistics; add a matching
`labels/` subfolder in YOLO format and it additionally computes real
mAP/precision/recall via Ultralytics' own validation path — the actual
number that would move Track B from "trained" to "validated" per the
roadmap's stated goal. This is the one roadmap item that isn't a code
problem: it's waiting on real photos, not on more engineering.
