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

        ttk.Checkbutton(stage1, text="No live preview (faster batch run)",
                         variable=self.dry_run_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 8))

        params = ttk.LabelFrame(stage1, text="Detection & tracking parameters", padding=8)
        params.grid(row=2, column=0, columnspan=3, sticky="we")
        params.columnconfigure(1, weight=1)

        ttk.Checkbutton(params, text="Auto-calibrate Min Area per video (recommended)",
                         variable=self.auto_min_area_var, command=self._toggle_min_area
                         ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(params, text="Min Area (px, juvenile cutoff):").grid(row=1, column=0, sticky="w", pady=2)
        self.min_area_spin = ttk.Spinbox(params, from_=0, to=3000, textvariable=self.min_area_var,
                                          width=8, state="disabled")
        self.min_area_spin.grid(row=1, column=1, sticky="w", pady=2)

        for i, (key, label, _default, lo, hi) in enumerate(PARAM_SPECS, start=2):
            ttk.Label(params, text=label + ":").grid(row=i, column=0, sticky="w", pady=2)
            ttk.Spinbox(params, from_=lo, to=hi, textvariable=self.param_vars[key],
                        width=8).grid(row=i, column=1, sticky="w", pady=2)

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
        self.status_var.set("Starting...")
        params = self.collect_params()
        self.worker = threading.Thread(
            target=self._work, args=(folder, self.dry_run_var.get(), params), daemon=True)
        self.worker.start()

    def _work(self, folder, dry_run, params):
        def progress(msg):
            self.queue.put(("status", msg))
        try:
            rows = core.run_batch(folder, "worm_motility_results.csv", dry_run, progress=progress, params=params)
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
                    if payload:
                        self.status_var.set(
                            f"Done — {len(payload)} worm tracks. CSV: worm_motility_results.csv")
                        self.show_results(payload)
                    else:
                        self.status_var.set("No videos found in that folder.")
                elif kind == "error":
                    self.run_btn.config(state="normal")
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

        self.tree = ttk.Treeview(stage3, columns=RESULT_COLUMNS, show="tree headings")
        self.tree.heading("#0", text="Video")
        self.tree.column("#0", width=260, anchor="w")
        for c in RESULT_COLUMNS:
            self.tree.heading(c, text=COLUMN_HEADERS[c])
            self.tree.column(c, width=95, anchor="center")
        scroll = ttk.Scrollbar(stage3, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def show_results(self, rows):
        by_video = {}
        for r in rows:
            by_video.setdefault(r["source_video"], []).append(r)

        for video, video_rows in by_video.items():
            healthy = sum(1 for r in video_rows if r["final_state"] == "HEALTHY")
            diseased = sum(1 for r in video_rows if r["final_state"] == "DISEASED")
            dead = sum(1 for r in video_rows if r["final_state"] == "DEAD")
            active = [r for r in video_rows if r["final_state"] in ("HEALTHY", "DISEASED")]
            avg_rate = sum(r["thrash_rate_per_min"] for r in active) / len(active) if active else 0.0
            parent_text = (f"{video}  —  {len(video_rows)} worms | "
                            f"Healthy:{healthy} Diseased:{diseased} Dead:{dead} | "
                            f"Avg {avg_rate:.1f}/min")
            parent = self.tree.insert("", "end", text=parent_text, open=True)
            for r in sorted(video_rows, key=lambda r: r["worm_id"]):
                self.tree.insert(parent, "end", text="", values=[r[c] for c in RESULT_COLUMNS])


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
