from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List

import clip
import onnxruntime as ort
import pandas as pd
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


@dataclass
class BenchmarkResult:
    times: int
    mean: float
    std: float

    def to_df(self, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame(
            data={
                "times": [self.times],
                "mean (ms)": [self.mean * 1000],
                "std (ms)": [self.std * 1000],
                **kwargs,
            }
        )

        return df

    def to_csv(self, filepath: Path, **kwargs):
        df = self.to_df(**kwargs)
        if filepath.exists():
            old_df = pd.read_csv(filepath)
            df = pd.concat([old_df, df])
            # df = df.reset_index()
        df.to_csv(filepath, index=False)


class Benchmark:
    def setup(self, batch_size: int):
        pass

    def run(self, n: int, batch_size: int):
        pass

    def __call__(self, n: int, batch_size: int) -> BenchmarkResult:
        self.setup(batch_size)
        # warmup
        for _ in range(4):
            self.run()
        torch.cuda.synchronize()
        # real benchmark
        times = []
        for _ in tqdm(range(n)):
            start = perf_counter()
            self.run()
            times.append(perf_counter() - start)

        torch.cuda.synchronize()
        times_t = torch.as_tensor(times)
        return BenchmarkResult(
            times=n, mean=times_t.mean().item(), std=times_t.std().item()
        )


class ClipTorchBenchmark(Benchmark):
    def __init__(self):
        self.model, _ = clip.load("ViT-B/32", jit=False, device="cuda")

    def setup(self, batch_size: int):
        self.image = torch.randn((batch_size, 3, 224, 224), device="cuda")

    @torch.no_grad()
    def run(self):
        self.model.encode_image(self.image)


class ClipTorchCompiledBenchmark(ClipTorchBenchmark):
    def __init__(self):
        super().__init__()
        self.model = torch.compile(self.model, mode="max-autotune")


class ClipOnnxBenchmark(Benchmark):
    def __init__(self):
        providers = ["CUDAExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            "ViT-B-32.onnx", sess_options=sess_options, providers=providers
        )

    def setup(self, batch_size: int):
        device_id = "cuda"
        x = torch.randn((batch_size, 3, 224, 224)).numpy()
        # see https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device
        image = ort.OrtValue.ortvalue_from_numpy(x, device_id, 0)
        output = ort.OrtValue.ortvalue_from_shape_and_type(
            [batch_size, 512], x.dtype, device_id, 0
        )
        self.io_binding = self.session.io_binding()
        self.io_binding.bind_ortvalue_input("image", image)
        self.io_binding.bind_ortvalue_output("output", output)

    def run(self):
        self.session.run_with_iobinding(self.io_binding)


if __name__ == "__main__":
    params = [
        dict(n=256, batch_size=1),
        dict(n=256, batch_size=4),
        dict(n=256, batch_size=8),
        dict(n=256, batch_size=16),
        dict(n=256, batch_size=32),
        dict(n=256, batch_size=64),
        dict(n=256, batch_size=128),
    ]

    benchmakrs: List[Benchmark] = [
        ("torch2.0", ClipTorchBenchmark()),
        ("torch2.0+compile", ClipTorchCompiledBenchmark()),
        ("onnx+cuda", ClipOnnxBenchmark()),
    ]

    for (name, benchmark) in benchmakrs:
        print(f"Running {name}", end=" ")
        for param in params:
            print(f"params = {param}")
            result = benchmark(**param)
            print(result.to_df())
            result.to_csv(
                Path("./results.csv"),
                name=name,
                n=param["n"],
                batch_size=param["batch_size"],
            )
            print("\n")
