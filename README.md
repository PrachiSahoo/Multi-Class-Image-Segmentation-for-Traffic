# U-Net Dashcam Segmentation

A U-Net (ResNet34 backbone) that segments lanes, pedestrians, and vehicles in urban driving dashcam footage — 19 classes total, following the standard Cityscapes/BDD100K taxonomy (see [classes.py](classes.py)).

This README is the complete guide from "just unzipped this" to "fully trained, running inference, and demoing it in a browser." Follow the steps in order.

> **Fastest way to see something working:** you don't need to train a model first. Do Step 1 (environment) and Step 2 (verify), then skip straight to Step 6 (web demo) — it runs on untrained weights out of the box and clearly labels itself as such. Training (Step 4) is only needed for the segmentation output to actually be *accurate*.

## Contents

- [Repository structure](#repository-structure)
- [Step 1: Set up the environment](#step-1-set-up-the-environment)
- [Step 2: Verify the setup works](#step-2-verify-the-setup-works)
- [Step 3: Add your dataset](#step-3-add-your-dataset)
- [Step 4: Train](#step-4-train)
- [Step 5: Run inference from the command line](#step-5-run-inference-from-the-command-line)
- [Step 6: Launch the interactive web demo](#step-6-launch-the-interactive-web-demo)
- [Step 7: Customize the frontend](#step-7-customize-the-frontend)
- [Completion roadmap](#completion-roadmap)
- [Troubleshooting](#troubleshooting)

## Repository structure

| Path | Purpose |
|---|---|
| `classes.py` | The 19-class label scheme, color palette, and class groupings |
| `model.py` | Builds the U-Net |
| `dataset.py` | Loads images/masks from disk, applies augmentations |
| `losses.py` | Combined Dice + Cross-Entropy loss |
| `train.py` | Training + validation loop, checkpointing |
| `inference.py` | Runs a trained model on an image or video from the command line |
| `utils.py` | Checkpoint save/load, mIoU metric, mask-to-color visualization |
| `test_pipeline.py` | End-to-end smoke test — no real data needed |
| `data/` | Where your dataset goes (Step 3) |
| `webapp/` | Local browser demo — Flask + HTML/CSS/JS (Step 6 & 7) |

## Step 1: Set up the environment

Do this once per machine.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Every time you come back to work on this later, just re-activate: `source venv/bin/activate` (or the Windows equivalent) — no need to reinstall.

Works on CUDA, Apple Silicon (MPS), or CPU-only — it auto-detects at runtime, no config needed.

## Step 2: Verify the setup works

```bash
python test_pipeline.py
```

This builds the model and runs one forward + backward pass on generated dummy data. It should end with `Pipeline test passed successfully!`.

No dataset is needed for this step — if it fails, the environment itself is broken and nothing past this point will work either. Fix this before moving on.

## Step 3: Add your dataset

```
data/
  train_images/   your training photos/frames
  train_masks/    matching label mask, same filename, .png extension
  val_images/     held-out set for tracking progress (optional but recommended)
  val_masks/      matching label mask
```

**Mask format:** a mask is a grayscale PNG the same size as its image, where each pixel's value is a class index (0–18), not a color:

- `0` = road
- `11` / `12` = person / rider
- `13`–`18` = vehicle types (car, truck, bus, train, motorcycle, bicycle)
- `255` = ignore this pixel

If your data is BDD100K or Cityscapes semantic segmentation labels, the shipped `trainId` masks already match this format exactly. Full matching rules (filenames, extensions) are in [data/README.md](data/README.md).

> If `dataset.py` raises `FileNotFoundError` when you start training, a mask is missing or its filename doesn't match its image — this is intentional, it fails loudly instead of silently training on a blank mask. Check the exact path in the error message.

### Where to get a dataset

Both of these ship `trainId` label files that match this project's mask format directly — no conversion needed.

| Dataset | Download | Best for |
|---|---|---|
| **BDD100K** (semantic segmentation) | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/) · download steps: [doc.bdd100k.com/download.html](https://doc.bdd100k.com/download.html) | Varied weather/time-of-day — closest match to "chaotic urban dashcam footage" |
| **Cityscapes** | [cityscapes-dataset.com/downloads](https://www.cityscapes-dataset.com/downloads/) | Grab `gtFine_trainvaltest.zip` (labels) + `leftImg8bit_trainvaltest.zip` (images) |

- Both require a free account before downloading.
- Use the label files named `*_trainId.png` — not the colored `*_color.png` visualizations.
- Sort the unzipped files into `data/train_images/`, `data/train_masks/`, `data/val_images/`, `data/val_masks/` per the layout above.

## Step 4: Train

```bash
python train.py
```

Hyperparameters live at the top of `train.py`:

| Setting | Default | Notes |
|---|---|---|
| `NUM_EPOCHS` | 5 | Raise this for a real run — 40–100+ is typical for semantic segmentation |
| `BATCH_SIZE` | 8 | Lower if you hit an out-of-memory error |
| `LEARNING_RATE` | 1e-4 | Reasonable starting point for AdamW on a pretrained encoder |
| `LOAD_MODEL` | False | Set `True` to resume from `unet_checkpoint_best.pth.tar` |

Each epoch prints train loss/mIoU, and — if `data/val_images/` has files — validation loss/mIoU too.

**Checkpoints saved:**
- `unet_checkpoint_epoch_N.pth.tar` — every epoch, unconditionally
- `unet_checkpoint_best.pth.tar` — only when validation mIoU improves (skipped entirely if there's no validation data)

mIoU (mean Intersection-over-Union) is the metric to watch: 0 is random, 1 is perfect. Early on, expect it to be low and noisy — a rising trend over epochs is what matters, not the absolute number on epoch 1.

Both `inference.py` (Step 5) and the web demo (Step 6) automatically pick up `unet_checkpoint_best.pth.tar` from the project root once it exists.

## Step 5: Run inference from the command line

```bash
# a single image
python inference.py --image path/to/photo.jpg --output result.jpg

# a video clip
python inference.py --image path/to/clip.mp4 --output result.mp4
```

Loads `unet_checkpoint_best.pth.tar` by default (override with `--model path/to/checkpoint.pth.tar`). Output is the original image/video with a colored segmentation overlay — class colors match `classes.py` (e.g. people are always reddish, vehicles bluish, road purple), so they're visually consistent frame to frame.

## Step 6: Launch the interactive web demo

`webapp/` is a local browser UI for the same model — upload a dashcam photo or clip and watch it get segmented. It's a thin wrapper around the existing pipeline: it imports `model.py`, `utils.py`, `dataset.py`, and `classes.py` directly, with no duplicated model logic.

**Features:**
- Before/after comparison slider for images, side-by-side original/segmented playback for video
- Live class-coverage breakdown for whatever was just processed
- Download button for the segmented result

**Run it:**
```bash
# with the main venv active (from Step 1)
pip install -r webapp/requirements.txt
python webapp/server.py
```

Then open **http://127.0.0.1:5050** in a browser.

- If `unet_checkpoint_best.pth.tar` doesn't exist yet, the app still runs — it serves from untrained weights and shows a clear warning banner, the same fallback behavior as `inference.py`.
- Video clips are processed frame-by-frame in the background with a live progress bar; a few seconds of footage works best so the wait stays short.
- Nothing under `webapp/` touches or imports `train.py`, so training and the demo are completely independent — run one, both, or neither.

## Step 7: Customize the frontend

Everything the browser loads lives in `webapp/static/` — plain HTML + CSS + vanilla JavaScript. No React/Vue/Svelte, no bundler, no `npm install`, no build step of any kind. The browser loads the files exactly as they sit on disk, so any AI coding assistant can read all three files in one pass and edit them directly, with zero project-specific tooling knowledge required.

### File map

| File | What it controls |
|---|---|
| `webapp/static/index.html` | Page structure/markup — sections, labels, layout order |
| `webapp/static/style.css` | All visual styling. Colors, spacing, and fonts are defined once as CSS variables at the top of the file (the `:root { ... }` block) — change `--accent`, `--bg`, `--surface`, etc. there to re-theme the whole page without hunting through the rest of it |
| `webapp/static/app.js` | Behavior — file upload, drag-and-drop, the compare slider, polling video-processing progress, talking to the backend |
| `webapp/server.py` | The Flask backend the frontend calls. Only touch this to change *what data* is available to the frontend, not how it looks |

Editing `index.html` / `style.css` / `app.js` just needs a browser refresh — no restart. Only editing `server.py` requires stopping and re-running `python webapp/server.py`.

### Current design

Useful context to hand an AI assistant before asking for a redesign — none of this is fixed, it's just the starting point:

- Dark, minimal theme — near-black background, one accent color (`--accent`, a soft blue)
- Sans-serif UI type; monospace reserved for numeric/technical readouts (inference time, resolution, percentages)
- A "viewfinder" motif — corner brackets on the upload area and result frame — nodding to the dashcam/vision-system subject matter
- Legend swatches use the model's *actual* class colors, not decorative ones

### Backend API contract

What `app.js` currently calls — give this to an AI assistant so it knows what data it can build UI around, or how to ask for new fields:

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/api/status` | GET | — | `{ device, checkpoint_loaded, checkpoint_path, num_classes }` |
| `/api/classes` | GET | — | `[{ name, color }]` — 19 entries, `color` is `#rrggbb`, sourced live from `classes.py` |
| `/api/segment` | POST | multipart field `image` | `{ overlay, coverage, inference_ms, device, checkpoint_loaded, width, height }` |
| `/api/segment-video` | POST | multipart field `video` | `{ job_id }` — starts background processing, does not block |
| `/api/segment-video/<job_id>/status` | GET | — | `{ status, progress, current_frame, total_frames, coverage, error, device }` |
| `/api/segment-video/<job_id>/result` | GET | — | The finished file, `Content-Type: video/mp4`, H.264-encoded |

- `overlay` is a ready-to-use `data:image/png;base64,...` string — drop it straight into an `<img src>`.
- `coverage` is `[{ name, percent }]`, sorted descending, only classes actually present in the result.
- `status` is `"processing"` / `"done"` / `"error"` — poll the status endpoint every ~700ms while `"processing"`.
- The result video plays directly in a `<video>` tag or downloads.

**Two implementation details** worth knowing before restyling those specific areas:
- The image before/after slider is a single CSS `clip-path` on `#img-overlay`, driven by the `--split` custom property; `app.js` updates that variable and the `.compare-handle` position together on slider input. Keep them in sync if you rebuild that area.
- Video processing runs on a background thread per upload, which is why the page polls `/status` instead of waiting on one long request — that's why there's a progress bar instead of a spinner for video.

### Handing this off to an AI assistant

Point it at this section plus the three files in `webapp/static/`. For a pure visual/UX redesign it shouldn't need to touch `server.py` or anything outside `webapp/` at all — the API contract above is everything it needs.

More detail: [webapp/README.md](webapp/README.md)

## Completion roadmap

The scaffold is functionally complete and tested, but a few things are worth doing before calling the project "done":

- **Train for real.** The default 5 epochs is a smoke-test value, not a real training budget.
- **Track experiments.** Right now you only get printed per-epoch numbers. If you're running multiple training configs, consider logging to TensorBoard or Weights & Biases so runs are comparable.
- **Per-class evaluation.** `calculate_iou` reports one averaged mIoU number. Before trusting the model on lanes/pedestrians/vehicles specifically, it's worth checking IoU broken out per class (or per group, using `ROAD_CLASS_IDS`/`PEDESTRIAN_CLASS_IDS`/`VEHICLE_CLASS_IDS` from `classes.py`) — a high average can still hide a model that's bad at the rare, small classes that matter most here.
- **Test on real, varied conditions.** Night driving, rain, glare, and occlusion are exactly the "chaotic" scenarios this project is meant for — validate against footage covering those, not just clean daytime frames.
- **Export for deployment**, if this needs to run outside this codebase (e.g. ONNX or TorchScript export, plus a latency check against your target frame rate).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: Could not read mask '...'` | A mask file is missing, or its filename doesn't match its image's basename — see Step 3 |
| `test_pipeline.py` fails | Environment/dependency issue — re-check Step 1 before touching any real data |
| Training runs but never prints `Val Loss` / `Val mIoU` | `data/val_images/` is empty — validation is skipped by design, add val files to enable it |
| `LOAD_MODEL = True` prints a warning and starts from scratch | `unet_checkpoint_best.pth.tar` doesn't exist yet — it's only created after at least one epoch completes with validation data present |
| Out-of-memory during training | Lower `BATCH_SIZE` in `train.py` |
| Web demo shows "no trained checkpoint" banner | Expected until `unet_checkpoint_best.pth.tar` exists in the project root (see Step 4) — the demo still works, just with untrained output |
| `Address already in use` when running `python webapp/server.py` | Another instance of the server is already running on port 5050 — stop it first, or change the port in the `app.run(...)` line at the bottom of `webapp/server.py` |
| Video upload in the web demo seems to hang | It's processing frame-by-frame in the background — watch the progress bar; longer clips take longer. Keep demo clips short (a few seconds) |
