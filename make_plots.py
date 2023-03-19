import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_DIR = "plots"

plt.style.use(["science"])
df = pd.read_csv("results/results.csv", index_col=None)
# printing to copy inside the readme
df[~df["cudnn_benchmark"]].to_markdown("results/torch-torch2-compile.md")
df[~df["do_compile"]].to_markdown("results/torch-cudnn_benchmark.md")


def make_plots_for_torch_vs_compile():
    for name in df.model_name.unique():
        print(name)
        curr_df = df[~df["cudnn_benchmark"]]
        curr_df = curr_df[curr_df["model_name"] == name]

        for image_size in df["image_size"].unique():
            plot_df = pd.DataFrame(
                data={
                    "torch2.0": curr_df[curr_df["image_size"] == image_size][
                        ~curr_df["do_compile"]
                    ]["mean"].values.tolist(),
                    "torch2.0+compile": curr_df[curr_df["image_size"] == image_size][
                        curr_df["do_compile"]
                    ]["mean"].values.tolist(),
                    "batch_size": curr_df[curr_df["image_size"] == image_size][
                        "batch_size"
                    ]
                    .unique()
                    .tolist(),
                }
            )
            plot_df = plot_df.set_index("batch_size")
            ax = plot_df.plot(y=["torch2.0", "torch2.0+compile"], kind="bar")
            plt.title(f"{name}, image_size={image_size}x{image_size}")
            ax.set_ylabel("Time (ms)")
            ax.set_xlabel("Batch Size")
            plt.gcf().savefig(
                f"{PLOT_DIR}/{name.replace('/', '-')}-{image_size}-results.jpeg",
                dpi=800,
            )


def make_plots_for_torch_vs_cudnn_benchmark():
    for name in df.model_name.unique():
        curr_df = df[~df["do_compile"]]
        curr_df = curr_df[curr_df["model_name"] == name]

        for image_size in df["image_size"].unique():
            plot_df = pd.DataFrame(
                data={
                    "torch2.0": curr_df[curr_df["image_size"] == image_size][
                        ~curr_df["cudnn_benchmark"]
                    ]["mean"].values.tolist(),
                    "torch2.0+cudnn_benchmark": curr_df[
                        curr_df["image_size"] == image_size
                    ][curr_df["cudnn_benchmark"]]["mean"].values.tolist(),
                    "batch_size": curr_df[curr_df["image_size"] == image_size][
                        "batch_size"
                    ]
                    .unique()
                    .tolist(),
                }
            )
            plot_df = plot_df.set_index("batch_size")
            ax = plot_df.plot(y=["torch2.0", "torch2.0+cudnn_benchmark"], kind="bar")
            plt.title(f"{name}, image_size={image_size}x{image_size}")
            ax.set_ylabel("Time (ms)")
            ax.set_xlabel("Batch Size")
            plt.gcf().savefig(
                f"{PLOT_DIR}/{name.replace('/', '-')}-{image_size}-cudnn_benchmark-results.jpeg",
                dpi=800,
            )


def print_stats():
    mean = df[~df["do_compile"]]["mean"].mean()
    compile_mean = df[df["do_compile"]]["mean"].mean()
    gain = (mean - compile_mean) / mean
    print(f"{gain} = {gain}")


# make_plots_for_torch_vs_compile()
make_plots_for_torch_vs_cudnn_benchmark()
# print_stats()
