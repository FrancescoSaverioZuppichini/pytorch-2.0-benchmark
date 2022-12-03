import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(["science"])

df = pd.read_csv("./results.csv", index_col=None)
del df["times"]
del df["n"]
# recreating the dataframe to idk make it work with the bar plot
# I suck guys
plot_df = pd.DataFrame(
    data={
        "torch2.0": df[df["name"] == "torch2.0"]["mean (ms)"].values.tolist(),
        "torch2.0+compile": df[df["name"] == "torch2.0+compile"][
            "mean (ms)"
        ].values.tolist(),
        "onnx+cuda": df[df["name"] == "onnx+cuda"]["mean (ms)"].values.tolist(),
        "batch_size": np.unique(df["batch_size"].values).tolist(),
    }
)
plot_df = plot_df.set_index("batch_size")
# printing to copy inside the readme
print(plot_df.to_markdown())
# print(plot_df)
ax = plot_df.plot(y=["torch2.0", "torch2.0+compile", "onnx+cuda"], kind="bar")
plt.title("PyTorch 2.0 vs ONNX on CLIP Image Encoder")
ax.set_ylabel("Time (ms)")
ax.set_xlabel("Batch Size")
plt.gcf().savefig("results.jpeg", dpi=800)
