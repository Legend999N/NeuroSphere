import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from neurosphere.config import ensure_directories, DEFAULT_TOP_K
from neurosphere.indexer import build_index
from neurosphere.search import SearchEngine


class NeuroSphereApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NeuroSphere")
        self.root.geometry("950x700")

        ensure_directories()

        self.search_engine = None

        self._build_ui()

    def _build_ui(self):

        title = ttk.Label(
            self.root,
            text="NeuroSphere",
            font=("Segoe UI", 22, "bold"),
        )

        title.pack(
            pady=(20, 5)
        )

        subtitle = ttk.Label(
            self.root,
            text="Local-first intelligent search",
        )

        subtitle.pack(
            pady=(0, 20)
        )

        controls = ttk.Frame(self.root)
        controls.pack(
            fill="x",
            padx=20,
        )

        self.folder_var = tk.StringVar()

        folder_entry = ttk.Entry(
            controls,
            textvariable=self.folder_var,
        )

        folder_entry.pack(
            side="left",
            fill="x",
            expand=True,
        )

        browse_button = ttk.Button(
            controls,
            text="Browse",
            command=self.browse_folder,
        )

        browse_button.pack(
            side="left",
            padx=5,
        )

        build_button = ttk.Button(
            controls,
            text="Build Index",
            command=self.build_index,
        )

        build_button.pack(
            side="left",
        )

        search_frame = ttk.Frame(self.root)
        search_frame.pack(
            fill="x",
            padx=20,
            pady=20,
        )

        self.query_var = tk.StringVar()

        query_entry = ttk.Entry(
            search_frame,
            textvariable=self.query_var,
            font=("Segoe UI", 12),
        )

        query_entry.pack(
            side="left",
            fill="x",
            expand=True,
        )

        query_entry.bind(
            "<Return>",
            lambda event: self.search(),
        )

        self.mode_var = tk.StringVar(
            value="Hybrid"
        )

        mode_box = ttk.Combobox(
            search_frame,
            textvariable=self.mode_var,
            values=[
                "BM25",
                "Dense",
                "Hybrid",
            ],
            state="readonly",
            width=12,
        )

        mode_box.pack(
            side="left",
            padx=5,
        )

        search_button = ttk.Button(
            search_frame,
            text="Search",
            command=self.search,
        )

        search_button.pack(
            side="left",
        )

        self.status_var = tk.StringVar(
            value="Build an index to begin."
        )

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
        )

        status.pack(
            anchor="w",
            padx=20,
        )

        self.results = tk.Text(
            self.root,
            wrap="word",
            font=("Consolas", 10),
        )

        self.results.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10,
        )

    def browse_folder(self):

        folder = filedialog.askdirectory()

        if folder:
            self.folder_var.set(folder)

    def build_index(self):

        folder = self.folder_var.get().strip()

        if not folder:
            messagebox.showwarning(
                "NeuroSphere",
                "Select a folder first.",
            )
            return

        try:

            self.status_var.set(
                "Building index. Please wait..."
            )

            self.root.update_idletasks()

            build_index(
                Path(folder),
                use_embeddings=True,
            )

            self.search_engine = SearchEngine()

            self.status_var.set(
                "Index built successfully."
            )

            messagebox.showinfo(
                "NeuroSphere",
                "Index built successfully.",
            )

        except Exception as exc:

            self.status_var.set(
                "Indexing failed."
            )

            messagebox.showerror(
                "Indexing Error",
                str(exc),
            )

    def search(self):

        query = self.query_var.get().strip()

        if not query:
            return

        if self.search_engine is None:

            try:
                self.search_engine = SearchEngine()

            except Exception:
                messagebox.showwarning(
                    "NeuroSphere",
                    "Build an index first.",
                )
                return

        mode = self.mode_var.get()

        try:

            if mode == "BM25":
                results = self.search_engine.bm25(
                    query,
                    DEFAULT_TOP_K,
                )

            elif mode == "Dense":
                results = self.search_engine.dense(
                    query,
                    DEFAULT_TOP_K,
                )

            else:
                results = self.search_engine.hybrid(
                    query,
                    DEFAULT_TOP_K,
                )

            self.display_results(
                query,
                mode,
                results,
            )

        except Exception as exc:

            messagebox.showerror(
                "Search Error",
                str(exc),
            )

    def display_results(
        self,
        query,
        mode,
        results,
    ):

        self.results.delete(
            "1.0",
            tk.END,
        )

        self.status_var.set(
            f"{len(results)} results"
        )

        self.results.insert(
            tk.END,
            f"Query: {query}\n"
        )

        self.results.insert(
            tk.END,
            f"Mode: {mode}\n"
        )

        self.results.insert(
            tk.END,
            "=" * 90
            + "\n\n"
        )

        if not results:

            self.results.insert(
                tk.END,
                "No results found.\n"
            )

            return

        for rank, result in enumerate(
            results,
            start=1,
        ):

            self.results.insert(
                tk.END,
                f"{rank}. {result['name']}\n"
            )

            self.results.insert(
                tk.END,
                f"   Path: {result['path']}\n"
            )

            self.results.insert(
                tk.END,
                f"   Score: {result['score']:.6f}\n"
            )

            snippet = result["content"]

            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            self.results.insert(
                tk.END,
                f"   {snippet}\n\n"
            )

    def close(self):
        if self.search_engine:
            self.search_engine.close()

        self.root.destroy()


def main():

    root = tk.Tk()

    app = NeuroSphereApp(root)

    root.protocol(
        "WM_DELETE_WINDOW",
        app.close,
    )

    root.mainloop()


if __name__ == "__main__":
    main()