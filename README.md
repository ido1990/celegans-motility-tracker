# celegans-motility-tracker

Zero-dependency-for-the-end-user desktop tool that batch-processes C. elegans
microscopy videos: detects worms, tracks each one frame-to-frame, filters out
juveniles/debris/dead worms, and computes a thrashing rate (body bends per
minute) per worm — with a live OpenCV preview and a CSV export.

## How it works

- **`analyzer.py`** — per-frame image analysis: builds a background model from
  the video, segments worms from it, classifies each contour as a juvenile,
  debris, a merged/occluded blob, or a candidate adult worm, and extracts a
  per-frame body-bend angle signal used to count thrashes.
- **`tracker.py`** — a centroid tracker (Hungarian assignment via
  `scipy.optimize.linear_sum_assignment`) that keeps a stable ID per worm
  across frames, tracks how long each worm stays visible, and flags a worm
  `DEAD` once both its position and body-bend deformation stay near-zero for
  a stretch of frames.
- **`gui.py`** — the app: opens each video in a folder, runs the pipeline,
  draws live overlays (bounding boxes, ID, status, thrash rate) with
  trackbars to tune detection on the fly, and writes results to a CSV.
- **`build.py`** — packages `gui.py` into a single portable executable via
  PyInstaller.

## Running from source

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python gui.py --folder assets
```

CLI options:

- `--folder PATH` — folder of videos to process (default: `assets`). Accepts
  `.avi`, `.mp4`, `.mov`, `.mkv`. Video files aren't tracked in this repo
  (one sample exceeds GitHub's 100MB file limit) — drop your own videos into
  `assets/` (or any folder) before running.
- `--output FILE` — CSV output path (default: `worm_motility_results.csv`).
- `--dry-run` — headless mode, no preview window (useful with no display, or
  for fast batch runs).

While the preview window is open:

- **q** or **Esc** — stop processing (still writes out results collected so far).
- Trackbars (read live every frame, no restart needed):
  - **Min Area** — contour area below this is classified as a juvenile.
  - **Sensitivity** — how large a body bend must be to count as a thrash
    (lower = counts smaller wiggles).
  - **Playback Speed (ms)** — delay between frames.
  - **Max Track Dist** — max pixel distance a worm can move between frames
    and still be matched to the same ID.
  - **Track Patience (fr)** — frames a worm can go undetected (e.g. behind
    another worm) before its ID is dropped.
  - **Healthy Rate** — thrash/min cutoff: at or above this, a tracked worm is
    labeled HEALTHY; below it, DISEASED.
  - **Dead Pos Delta** / **Dead Bend Delta** / **Dead Window (fr)** — how
    little a worm's position/body-bend may vary, and over how many frames,
    before it's flagged DEAD.

### On Windows, without installing anything manually

Double-click **`run_windows.bat`** (or run it from a terminal). It creates a
local virtual environment, installs dependencies, and launches the tool. You
still need Python installed from [python.org](https://www.python.org/) with
"Add python.exe to PATH" checked during setup — this is the fastest way to
test the tool on Windows without building an executable.

## Output

`worm_motility_results.csv`, one row per tracked worm per video:

| column | meaning |
|---|---|
| `source_video` | file the row came from |
| `worm_id` | tracker-assigned ID (not stable across videos) |
| `final_state` | `HEALTHY`, `DISEASED`, or `DEAD` |
| `frame_entry` / `frame_exit` | first/last frame the worm was tracked |
| `visible_frames` | frames the worm was actually tracked (excludes offscreen time) |
| `total_thrashes` | body bends counted over the tracked period |
| `thrash_rate_per_min` | `total_thrashes / (visible_frames / fps) * 60` |
| `mean_area` | average contour area in pixels |
| `mean_bend_amplitude_deg` | average body-bend magnitude |

## Self-checks

`analyzer.py` and `tracker.py` each have a small built-in check (synthetic
shapes / synthetic tracking scenarios, plain asserts, no test framework):

```bash
python analyzer.py
python tracker.py
```

## Building a standalone executable

```bash
pip install -r requirements.txt
python build.py
```

Produces a single-file executable in `dist/` (`MotilityTracker` /
`MotilityTracker.exe`) that end users can run without installing Python or
any dependency.

**PyInstaller builds are OS-specific — it does not cross-compile.** Building
on Linux produces a Linux binary; to get a Windows `.exe` for non-technical
users, run `build.py` on an actual Windows machine (or a Windows CI runner).
There is no reliable way to produce a genuine Windows `.exe` from Linux for
an OpenCV/SciPy-heavy app (Wine-based cross-builds are fragile and not used
here) — use `run_windows.bat` above to test on Windows via Python instead,
and build the real `.exe` from Windows when you're ready to distribute it.

`build.py` explicitly excludes a few unrelated heavy packages
(`torch`, `sklearn`, `nvidia`, `triton`, `jax`, `cupy`) that PyInstaller can
otherwise pull in if they happen to be installed on the build machine, due to
an optional SciPy compatibility shim that references them — none of this
project's code imports them.

## Known limitations

- The tracker is a simple centroid/Hungarian matcher, not a full multi-object
  re-identification system. In dense footage with lots of worm-on-worm
  occlusion, a worm's ID can occasionally churn (dropped and re-registered as
  a new ID) rather than being perfectly preserved across every crossing.
  `Max Track Dist` and `Track Patience` are the main levers to tune this for
  your footage.
