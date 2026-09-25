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
- **`gui.py`** — the processing engine + CLI: opens each video in a folder,
  runs the pipeline, draws live overlays (bounding boxes, ID, status, thrash
  rate) with trackbars to tune detection on the fly, and writes results to a
  CSV. Exposes `run_batch()` for other front-ends (like `launcher.py`) to
  call directly.
- **`launcher.py`** — **this is the app end users run.** One window, three
  stacked stages:
  1. **Folder & Parameters** — pick a folder, and set every detection/tracking
     knob (Sensitivity, Healthy Rate, Max Track Dist, Track Patience, Dead
     Pos/Bend Delta, Dead Window, Playback Speed) before running, instead of
     dragging cv2 trackbars mid-playback. Min Area has an "Auto-calibrate"
     checkbox (on by default, one value per video — see below) or can be
     pinned to one fixed number for the whole batch.
     Hover any parameter label to see a tooltip explaining what it does.
  2. **Run** — processes every video in a background thread so the window
     stays responsive; the live OpenCV preview (unless "no live preview" is
     checked) still pops up during processing, using the parameters you set
     as their starting point — you can still drag the cv2 trackbars live if
     you want to fine-tune mid-run. A **Stop** button next to Run cancels an
     in-progress run — checked once per frame (and during the per-video
     calibration phase), so it responds within about a second — and keeps
     whatever results were already collected.
  3. **Results** — a table, grouped by video with a summary line per video
     (worm counts by state and the average thrash rate among that video's
     HEALTHY worms), appears in the same window when the run finishes. No
     need to open the CSV, though it's still written alongside.
- **`build.py`** — packages `launcher.py` into a single portable executable
  via PyInstaller.

### Matching the original requirements

- **Dual strain (CL2122 healthy / GMC diseased) handling** — segmentation
  uses a self-adapting threshold (Otsu on a background-subtracted frame,
  recomputed per video) so it isn't tuned to one strain's contrast or
  lighting condition. Motion pace differs a lot between strains, so *what
  counts as fast enough to be "healthy"* is the **Healthy Rate** trackbar
  (thrashes/min cutoff) — turn it down for a GMC (diseased, slow) video, up
  for a CL2122 (healthy, fast) one.
- **Lighting** — already auto-adjusted per video: the segmentation threshold
  isn't a fixed brightness value, it's an Otsu split computed fresh on each
  video's own background-subtracted frames, so a darker or brighter
  recording doesn't need retuning.
- **Only adult worms counted** — `classify_contour()` filters out contours
  below **Min Area** as juveniles (recently hatched, small) before they ever
  reach the tracker; they're drawn gray, never tracked, never counted.
  **Min Area auto-calibrates per video** (`analyzer.estimate_min_area`):
  before processing starts, it samples ~25 frames from that video, collects
  every contour's area, and splits the distribution into a small
  (juvenile/debris) and large (adult) population via Otsu on the log-scaled
  areas — the same technique already used for segmentation, just applied to
  area instead of pixel intensity. This means the same physical worm size
  doesn't need a different hand-typed number at every zoom/magnification —
  the trackbar starts at the calibrated value and prints it to the console
  (`auto-calibrated Min Area for <video>: <N>px`); you can still drag it if
  the auto value is wrong for unusual footage.
- **Dead worms excluded from the count** — a track only flips to `DEAD` once
  *both* its position and body-bend angle stay within a small delta (**Dead
  Pos Delta** / **Dead Bend Delta**) for a window of frames (**Dead Window**),
  and `DEAD` tracks are excluded from `Avg Thrashes/Min` and from the
  per-video HEALTHY average shown in the results window.
- **Worms entering/leaving frame, collisions** — `thrash_rate_per_min` is
  computed as `total_thrashes / (measured_frames / fps) * 60`, where
  `measured_frames` counts only the frames in which that worm's body shape
  was actually measured — not the full video, and not stretches where it was
  merged with another worm or briefly lost (gaps of up to 3 frames are
  bridged). Tracks measured for less than 5 s are dropped from the results:
  they're mostly fragments of a track broken by a collision.
- **Average thrashes/min** — shown live during processing (`Avg
  Thrashes/Min` overlay, active worms only) and per video in the results
  window after a run finishes.

## Running from source

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python launcher.py
```

For scripting/automation without the GUI, `gui.py` still works standalone:

```bash
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

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/) —
   during setup, check **"Add python.exe to PATH"**.
2. Download/copy this project folder onto the Windows machine.
3. Double-click **`run_windows.bat`** (or open a terminal in the folder and
   run `run_windows.bat`).

What it does, each time you run it:

- Creates a local virtual environment in `venv\` (first run only — reused
  after that).
- Installs/updates everything in `requirements.txt` into that venv.
- Launches `launcher.py` — the folder-picker GUI window should appear.
- Leaves a terminal window open (`pause` at the end) so you can read any
  error output if something went wrong.

This is the fastest way to run/test the tool on Windows — no compiling, no
admin rights beyond a normal Python install. Re-run the `.bat` any time; it
won't recreate the venv unless you delete the `venv\` folder.

**Troubleshooting:**
- `'python' is not recognized...` — Python isn't on PATH; reinstall from
  python.org with the PATH checkbox checked, or run `py run_windows.bat`
  equivalent commands manually with the `py` launcher instead.
- Antivirus/SmartScreen flags the `.bat` or the built `.exe` — expected for
  an unsigned script/executable from a new source; choose "Run anyway" /
  "More info → Run anyway".

## Output

`worm_motility_results.csv`, one row per tracked worm per video:

| column | meaning |
|---|---|
| `source_video` | file the row came from |
| `worm_id` | tracker-assigned ID (not stable across videos) |
| `final_state` | `HEALTHY`, `DISEASED`, or `DEAD` |
| `frame_entry` / `frame_exit` | first/last frame the worm was tracked |
| `visible_frames` | frames the worm was actually tracked (excludes offscreen time) |
| `measured_frames` | frames within that where its body shape was measured (excludes collisions) |
| `total_thrashes` | body bends counted over the measured frames |
| `thrash_rate_per_min` | `total_thrashes / (measured_frames / fps) * 60` |
| `mean_area` | average contour area in pixels |
| `mean_bend_amplitude_deg` | average body-bend magnitude |
| `centerline_measured_frames` / `centerline_thrashes` / `thrash_rate_centerline` | experimental second method: thrashes from the signed bend of the worm's centerline (see below) |

## Validation against hand counts

`validation/manual_counts_2026-09-03.csv` holds hand-counted thrash rates (78
worms, 8 videos: CL2122 / GMC101 × ddw / 300ul np). `validation/compare.py`
runs the pipeline on those videos and compares per-video mean rates (hand
counts cover a subset of worms with no positions, so worms aren't paired
one-to-one):

```bash
python validation/compare.py --videos <folder with the 03.09.26 videos>
```

Current result: mean absolute error 7.1 thrashes/min per video (was 25.6
before rates were normalized by measured time and Sensitivity was retuned
from 15 to 12). The remaining error is mostly CL2122 in 300ul np, which is
undercounted by 13–18/min.

### Per-worm validation

Per-video averages can hide per-worm errors, so the next validation round is
per worm:

1. `python validation/annotate.py --videos <folder> --out <folder>` renders
   each video with the tracker's worm IDs (only worms it reports) and saves
   their positions (`validation/tracks/`) so counts stay matchable if the
   numbering ever changes.
2. Count numbered worms on those videos. The lab's counting page does this
   with a tap counter and records each count's exact video window.
3. `python validation/compare_worms.py --videos <folder> --counts <csv>`
   scores every counting method on exactly those worms and windows.

Two methods are computed side by side. `thrash_rate_per_min` (default) counts
peaks of the unsigned mid-body angle. `thrash_rate_centerline` traces the
worm's centerline, measures the signed bend between its head and tail halves,
and counts full left-right swings; frames whose centerline fails quality checks
(length, symmetric outline, no impossible jumps) are skipped. The centerline
signal is much cleaner on well-separated worms and fixes the fast CL2122 video
(16-30-15-574: -17.5 to -1.1), but on per-video averages it is not yet better
overall (7.7 vs 7.1), so it stays experimental until per-worm counts decide.

## Self-checks

`analyzer.py` and `tracker.py` each have a small built-in check (synthetic
shapes / synthetic tracking scenarios, plain asserts, no test framework):

```bash
python analyzer.py
python tracker.py
```

## Building a standalone executable

**PyInstaller builds are OS-specific — it does not cross-compile.** Building
on Linux produces a Linux binary, building on macOS produces a macOS binary,
building on Windows produces a Windows `.exe`. There's no reliable way to
produce a genuine Windows `.exe` from Linux/macOS for an OpenCV/SciPy-heavy
app (Wine-based cross-builds are fragile and not used here) — to distribute
a `.exe`, run the build on an actual Windows machine (or a `windows-latest`
CI runner).

The build command is the same everywhere:

```bash
pip install -r requirements.txt
python build.py
```

`build.py` explicitly excludes a few unrelated heavy packages
(`torch`, `sklearn`, `nvidia`, `triton`, `jax`, `cupy`) that PyInstaller can
otherwise pull in if they happen to be installed on the build machine, due to
an optional SciPy compatibility shim that references them — none of this
project's code imports them.

### Windows

```bat
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python build.py
```

Produces `dist\MotilityTracker.exe`. Copy that one file anywhere on another
Windows machine and double-click it — no Python install needed there.

### macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python build.py
```

Produces `dist/MotilityTracker`. On first launch, Gatekeeper will likely
block it as from an unidentified developer — right-click → **Open** once to
approve it (or `xattr -d com.apple.quarantine dist/MotilityTracker`).

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python build.py
```

Produces `dist/MotilityTracker`. `chmod +x dist/MotilityTracker` if it isn't
already executable, then run it directly (`./dist/MotilityTracker`).

All three produce a single-file executable that end users can run without
installing Python or any dependency — just not portable *across* platforms,
so build once per target OS.

## Known limitations

- The tracker is a simple centroid/Hungarian matcher, not a full multi-object
  re-identification system. In dense footage with lots of worm-on-worm
  occlusion, a worm's ID can occasionally churn (dropped and re-registered as
  a new ID) rather than being perfectly preserved across every crossing.
  `Max Track Dist` and `Track Patience` are the main levers to tune this for
  your footage — the defaults (55px / 30 frames) were raised from an earlier,
  stricter baseline (40px / 18 frames) after measuring on the sample videos
  that the looser settings cut total track-ID churn by roughly a third
  without visibly worse mismatches; raise `Track Patience` further for very
  dense, slow-moving footage, or lower both back toward the old defaults if
  you see IDs jumping between worms that are close together. When two worms
  merge into one blob, the tracker no longer
  lets that blob's blended centroid drag either worm's tracked position
  toward the other — each frozen track only resumes moving once it's seen on
  its own again — which avoids one source of position corruption during an
  overlap, but genuinely disambiguating *which* worm went which way through
  a crossing would need appearance-based re-identification, which this
  project intentionally doesn't implement (see the design tradeoffs above).
