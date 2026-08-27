"""Tkinter front-end: pick a folder, hit Run, see per-worm results without opening the CSV.

This is the entry point end users run (and what build.py packages). The processing
pipeline itself lives in gui.py / analyzer.py / tracker.py; this file is UI-only.
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import gui as core

RESULT_COLUMNS = ["worm_id", "final_state", "frame_entry", "frame_exit", "visible_frames",
                   "total_thrashes", "thrash_rate_per_min", "mean_area", "mean_bend_amplitude_deg"]
COLUMN_HEADERS = {
    "worm_id": "ID", "final_state": "State", "frame_entry": "In", "frame_exit": "Out",
    "visible_frames": "Visible Fr.", "total_thrashes": "Thrashes",
    "thrash_rate_per_min": "Thrash/min", "mean_area": "Mean Area",
    "mean_bend_amplitude_deg": "Mean Bend deg",
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("C. elegans Motility Tracker")

        self.folder_var = tk.StringVar(value="assets")
        self.dry_run_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")
        self.queue = queue.Queue()
        self.worker = None

        frm = ttk.Frame(root, padding=12)
        frm.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Video folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.folder_var, width=50).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="Browse...", command=self.browse).grid(row=0, column=2)

        ttk.Checkbutton(frm, text="No live preview (faster batch run)",
                         variable=self.dry_run_var).grid(row=1, column=1, sticky="w", pady=4)

        self.run_btn = ttk.Button(frm, text="Run", command=self.run)
        self.run_btn.grid(row=2, column=1, pady=8)

        ttk.Label(frm, textvariable=self.status_var, foreground="#555").grid(
            row=3, column=0, columnspan=3, sticky="w")

        self.root.after(150, self._poll_queue)

    def browse(self):
        path = filedialog.askdirectory(initialdir=self.folder_var.get() or ".")
        if path:
            self.folder_var.set(path)

    def run(self):
        if self.worker and self.worker.is_alive():
            return
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Choose a valid folder first.")
            return
        self.run_btn.config(state="disabled")
        self.status_var.set("Starting...")
        self.worker = threading.Thread(
            target=self._work, args=(folder, self.dry_run_var.get()), daemon=True)
        self.worker.start()

    def _work(self, folder, dry_run):
        def progress(msg):
            self.queue.put(("status", msg))
        try:
            rows = core.run_batch(folder, "worm_motility_results.csv", dry_run, progress=progress)
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

    def show_results(self, rows):
        win = tk.Toplevel(self.root)
        win.title("Results")
        win.geometry("950x520")

        tree = ttk.Treeview(win, columns=RESULT_COLUMNS, show="tree headings")
        tree.heading("#0", text="Video")
        tree.column("#0", width=260, anchor="w")
        for c in RESULT_COLUMNS:
            tree.heading(c, text=COLUMN_HEADERS[c])
            tree.column(c, width=95, anchor="center")
        scroll = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(fill="both", expand=True, side="left")
        scroll.pack(side="right", fill="y")

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
            parent = tree.insert("", "end", text=parent_text, open=True)
            for r in sorted(video_rows, key=lambda r: r["worm_id"]):
                tree.insert(parent, "end", text="", values=[r[c] for c in RESULT_COLUMNS])


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
