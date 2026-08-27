"""Tkinter front-end: one window, three stages — set folder & parameters, hit Run, see
results — without opening the CSV or fumbling with cv2 trackbars mid-playback.

This is the entry point end users run (and what build.py packages). The processing
pipeline itself lives in gui.py / analyzer.py / tracker.py; this file is UI-only.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import gui as core
import tracker as tracker_mod

RESULT_COLUMNS = ["worm_id", "final_state", "frame_entry", "frame_exit", "visible_frames",
                   "total_thrashes", "thrash_rate_per_min", "mean_area", "mean_bend_amplitude_deg"]
COLUMN_HEADERS = {
    "worm_id": "ID", "final_state": "State", "frame_entry": "In", "frame_exit": "Out",
    "visible_frames": "Visible Fr.", "total_thrashes": "Thrashes",
    "thrash_rate_per_min": "Thrash/min", "mean_area": "Mean Area",
    "mean_bend_amplitude_deg": "Mean Bend deg",
}
# Per-video summary stats, shown only on each video's parent row (blank on worm rows) — kept as
# real columns rather than packed into the tree label, since that column is too narrow to show them.
SUMMARY_COLUMNS = ["avg_all_rate", "avg_healthy_rate"]
SUMMARY_HEADERS = {"avg_all_rate": "Avg All/min", "avg_healthy_rate": "Avg Healthy/min"}

# key, label, default, min, max — mirrors gui.py's cv2 trackbars, minus Min Area (handled
# separately below since it can be auto-calibrated per video instead of a fixed number).
PARAM_SPECS = [
    ("sensitivity", "Sensitivity (thrash prominence, deg)", core.DEFAULT_SENSITIVITY, 1, 60),
    ("healthy_threshold", "Healthy Rate (thrashes/min cutoff)", core.DEFAULT_HEALTHY_THRESHOLD, 0, 100),
    ("max_distance", "Max Track Dist (px)", int(tracker_mod.MAX_DISTANCE), 1, 150),
    ("max_disappeared", "Track Patience (frames)", tracker_mod.MAX_DISAPPEARED, 1, 120),
    ("dead_pos_delta", "Dead Pos Delta (px)", int(tracker_mod.DEAD_POSITION_DELTA), 0, 60),
    ("dead_bend_delta", "Dead Bend Delta (deg)", int(tracker_mod.DEAD_BEND_DELTA), 0, 90),
    ("dead_window", "Dead Window (frames)", tracker_mod.DEAD_WINDOW_FRAMES, 2, 300),
    ("delay_ms", "Playback Speed (ms, live preview only)", core.DEFAULT_DELAY_MS, 1, 200),
]

PARAM_TOOLTIPS = {
    "sensitivity": "Minimum body-bend angle change (deg) to count as one thrash. Lower = more "
                   "sensitive (catches subtler bends but more noise); higher = only strong bends count.",
    "healthy_threshold": "Thrash rate (thrashes/min) cutoff. A worm that's alive (not DEAD) is "
                          "labeled HEALTHY at or above this rate, DISEASED below it.",
    "max_distance": "Max pixel distance a worm can move between frames and still match its "
                     "existing track. Raise for fast movers or low frame rate; lower to cut down "
                     "ID swaps between worms passing near each other.",
    "max_disappeared": "How many consecutive frames a worm can go undetected (e.g. hidden behind "
                        "another worm) before its track is ended. Higher survives longer occlusions "
                        "but risks the track inheriting the wrong worm's identity afterward.",
    "dead_pos_delta": "Max total position drift (px) allowed over the Dead Window for a worm to "
                       "be classified DEAD, rather than just briefly holding still.",
    "dead_bend_delta": "Max body-bend-angle drift (deg) allowed over the Dead Window for a worm "
                        "to be classified DEAD.",
    "dead_window": "Number of frames over which position/bend drift is measured to decide if a "
                   "worm has stopped moving for good (DEAD) or is just briefly still.",
    "delay_ms": "Delay (ms) between frames in the live preview window. Higher = slower playback. "
                "Only affects the live preview, not batch processing speed.",
}
AUTO_MIN_AREA_TOOLTIP = ("Automatically picks the Min Area cutoff from each video's own footage "
                          "instead of one fixed value — recommended when videos are shot at "
                          "different zoom levels.")
MIN_AREA_TOOLTIP = ("Minimum contour area (px) to count as an adult worm. Smaller blobs are "
                     "classified as juveniles and excluded from tracking/thrash counting.")
DRY_RUN_TOOLTIP = ("Skip the live preview window and run as fast as possible — recommended for "
                    "batch processing many videos unattended.")


class Tooltip:
    """Minimal hover tooltip: a borderless Toplevel label shown on <Enter>, hidden on <Leave>."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        ttk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid",
                  borderwidth=1, padding=4, wraplength=280, justify="left").pack()

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("C. elegans Motility Tracker")
        root.geometry("980x760")
        root.minsize(760, 560)
        root.columnconfigure(0, weight=1)

        self.folder_var = tk.StringVar(value="assets")
        self.dry_run_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")
        self.auto_min_area_var = tk.BooleanVar(value=True)
        self.min_area_var = tk.IntVar(value=core.DEFAULT_MIN_AREA)
        self.param_vars = {key: tk.IntVar(value=default) for key, _label, default, _lo, _hi in PARAM_SPECS}
        self.queue = queue.Queue()
        self.worker = None
        self.stop_event = None

        self._build_stage1_folder_and_params()
        self._build_stage2_run()
        self._build_stage3_results()

        self.root.after(150, self._poll_queue)

    # ---- Stage 1: folder + parameters ----

    def _build_stage1_folder_and_params(self):
        stage1 = ttk.LabelFrame(self.root, text="1. Folder & Parameters", padding=10)
        stage1.grid(row=0, column=0, sticky="we", padx=10, pady=(10, 4))
        stage1.columnconfigure(1, weight=1)

        ttk.Label(stage1, text="Video folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(stage1, textvariable=self.folder_var).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(stage1, text="Browse...", command=self.browse).grid(row=0, column=2)

        dry_run_check = ttk.Checkbutton(stage1, text="No live preview (faster batch run)",
                                          variable=self.dry_run_var)
        dry_run_check.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 8))
        Tooltip(dry_run_check, DRY_RUN_TOOLTIP)

        params = ttk.LabelFrame(stage1, text="Detection & tracking parameters", padding=8)
        params.grid(row=2, column=0, columnspan=3, sticky="we")
        params.columnconfigure(1, weight=1)

        auto_min_area_check = ttk.Checkbutton(
            params, text="Auto-calibrate Min Area per video (recommended)",
            variable=self.auto_min_area_var, command=self._toggle_min_area)
        auto_min_area_check.grid(row=0, column=0, columnspan=3, sticky="w")
        Tooltip(auto_min_area_check, AUTO_MIN_AREA_TOOLTIP)

        min_area_label = ttk.Label(params, text="Min Area (px, juvenile cutoff):")
        min_area_label.grid(row=1, column=0, sticky="w", pady=2)
        self.min_area_spin = ttk.Spinbox(params, from_=0, to=3000, textvariable=self.min_area_var,
                                          width=8, state="disabled")
        self.min_area_spin.grid(row=1, column=1, sticky="w", pady=2)
        Tooltip(min_area_label, MIN_AREA_TOOLTIP)
        Tooltip(self.min_area_spin, MIN_AREA_TOOLTIP)

        for i, (key, label, _default, lo, hi) in enumerate(PARAM_SPECS, start=2):
            tooltip_text = PARAM_TOOLTIPS[key]
            param_label = ttk.Label(params, text=label + ":")
            param_label.grid(row=i, column=0, sticky="w", pady=2)
            param_spin = ttk.Spinbox(params, from_=lo, to=hi, textvariable=self.param_vars[key], width=8)
            param_spin.grid(row=i, column=1, sticky="w", pady=2)
            Tooltip(param_label, tooltip_text)
            Tooltip(param_spin, tooltip_text)

    def _toggle_min_area(self):
        self.min_area_spin.config(state="disabled" if self.auto_min_area_var.get() else "normal")

    def browse(self):
        path = filedialog.askdirectory(initialdir=self.folder_var.get() or ".")
        if path:
            self.folder_var.set(path)

    def collect_params(self):
        params = {key: var.get() for key, var in self.param_vars.items()}
        if not self.auto_min_area_var.get():
            params["min_area"] = self.min_area_var.get()
        return params

    # ---- Stage 2: run ----

    def _build_stage2_run(self):
        stage2 = ttk.Frame(self.root, padding=(10, 4))
        stage2.grid(row=1, column=0, sticky="we")
        self.run_btn = ttk.Button(stage2, text="2. Run", command=self.run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(stage2, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))
        ttk.Label(stage2, textvariable=self.status_var, foreground="#555").pack(side="left", padx=10)

    def run(self):
        if self.worker and self.worker.is_alive():
            return
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Choose a valid folder first.")
            return
        self.tree.delete(*self.tree.get_children())
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Starting...")
        params = self.collect_params()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(
            target=self._work, args=(folder, self.dry_run_var.get(), params, self.stop_event), daemon=True)
        self.worker.start()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
            self.status_var.set("Stopping...")
            self.stop_btn.config(state="disabled")

    def _work(self, folder, dry_run, params, stop_event):
        def progress(msg):
            self.queue.put(("status", msg))
        try:
            rows = core.run_batch(folder, "worm_motility_results.csv", dry_run,
                                   progress=progress, params=params, stop_event=stop_event)
            self.queue.put(("done", rows))
        except Exception as e:
            self.queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if payload:
                        self.status_var.set(
                            f"Done — {len(payload)} worm tracks. CSV: worm_motility_results.csv")
                        self.show_results(payload)
                    elif self.stop_event is not None and self.stop_event.is_set():
                        self.status_var.set("Stopped — no results yet.")
                    else:
                        self.status_var.set("No videos found in that folder.")
                elif kind == "error":
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.status_var.set("Error.")
                    messagebox.showerror("Processing failed", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    # ---- Stage 3: results ----

    def _build_stage3_results(self):
        stage3 = ttk.LabelFrame(self.root, text="3. Results", padding=10)
        stage3.grid(row=2, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.root.rowconfigure(2, weight=1)
        stage3.rowconfigure(0, weight=1)
        stage3.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(stage3, columns=RESULT_COLUMNS + SUMMARY_COLUMNS, show="tree headings")
        self.tree.heading("#0", text="Video")
        self.tree.column("#0", width=260, anchor="w")
        for c in RESULT_COLUMNS:
            self.tree.heading(c, text=COLUMN_HEADERS[c])
            self.tree.column(c, width=95, anchor="center")
        for c in SUMMARY_COLUMNS:
            self.tree.heading(c, text=SUMMARY_HEADERS[c])
            self.tree.column(c, width=105, anchor="center")
        vscroll = ttk.Scrollbar(stage3, orient="vertical", command=self.tree.yview)
        hscroll = ttk.Scrollbar(stage3, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="we")

    def show_results(self, rows):
        by_video = {}
        for r in rows:
            by_video.setdefault(r["source_video"], []).append(r)

        for video, video_rows in by_video.items():
            healthy_rows = [r for r in video_rows if r["final_state"] == "HEALTHY"]
            diseased = sum(1 for r in video_rows if r["final_state"] == "DISEASED")
            dead = sum(1 for r in video_rows if r["final_state"] == "DEAD")
            avg_all_rate = sum(r["thrash_rate_per_min"] for r in video_rows) / len(video_rows)
            avg_healthy_rate = (sum(r["thrash_rate_per_min"] for r in healthy_rows) / len(healthy_rows)
                                 if healthy_rows else 0.0)
            parent_text = (f"{video}  —  {len(video_rows)} worms | "
                            f"Healthy:{len(healthy_rows)} Diseased:{diseased} Dead:{dead}")
            summary_values = [""] * len(RESULT_COLUMNS) + [f"{avg_all_rate:.1f}", f"{avg_healthy_rate:.1f}"]
            parent = self.tree.insert("", "end", text=parent_text, open=True, values=summary_values)
            for r in sorted(video_rows, key=lambda r: r["worm_id"]):
                self.tree.insert(parent, "end", text="",
                                  values=[r[c] for c in RESULT_COLUMNS] + ["", ""])


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
