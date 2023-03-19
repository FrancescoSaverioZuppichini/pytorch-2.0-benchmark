MODEL_NAME = "ViT-L/14"
import torch
from torch.utils import benchmark
import pandas as pd
from functools import partial
import clip
from clip.model import VisionTransformer
from torch import nn
from typing import Tuple
from torchvision.models import resnet34
from torchvision.models import resnext50_32x4d

from pprint import pprint

RESULTS_PATH = "results/results.csv"


def get_clip_vision(name: str = "ViT-B/32"):

    model, preprocess = clip.load(name, device="cuda")
    # this runs by default on mixed fp16
    model_vision = model.visual

    return model_vision


def get_model(
    name: str,
):
    zoo = {
        "clip_vision_vit-b/32": partial(
            get_clip_vision,
            "ViT-B/32",
        ),
        "clip_vision_vit-l/14": partial(
            get_clip_vision,
            "ViT-L/14",
        ),
        "clip_vision_vit-rn50": partial(
            get_clip_vision,
            "RN50",
        ),
        "resnet34": resnet34,
        "resnext50_32x4d": resnext50_32x4d,
    }

    return zoo[name]()


def get_run_function(
    model: nn.Module,
    batch_size: int,
    image_size: Tuple[int, int] = (224, 224),
    is_fp16: bool = True,
    do_compile: bool = False,
):

    model = model.cuda().eval()

    if do_compile:
        print("[INFO] compiling")
        model = torch.compile(model, mode="max-autotune")

    x = torch.randn((batch_size, 3, *image_size), device="cuda")

    if is_fp16:
        print("[INFO] using fp16")
        x = x.half()
        if isinstance(model, VisionTransformer):
            # clip uses some sort of mixed fp16 by defaul
            print("[DEBUG] Clip already in fp16")
        else:
            model = model.half()
    print(f"[INFO] using input with shape = {x.shape}")

    @torch.no_grad()
    def _run():
        model(x)

    return _run


def _wrap_for_n_times(fn, n: int):
    for _ in range(n):
        fn()


def profile_model(fn, min_run_time=30):
    # warmup
    _wrap_for_n_times(fn, 4)
    print("[INFO] profiling...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    res = benchmark.Timer(
        stmt=f"fn()",
        globals={"_wrap_for_n_times": _wrap_for_n_times, "fn": fn},
        label="profile",
        sub_label="",
        description="",
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2**20
    memory_mb = f"Memory used: {memory} MB"
    print(f"[INFO] time={res} memory={memory_mb}")
    return res.mean * 1000, res.median * 1000, res.number_per_run, memory_mb


def save_to_csv(**kwargs):
    df = pd.DataFrame.from_records([kwargs])
    df.to_csv(RESULTS_PATH, mode="a", index=False, float_format="%.2f")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark a model and store the results."
    )
    parser.add_argument(
        "model_name", type=str, help="name of the machine learning model to run"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for the model (default: 32)",
    )
    parser.add_argument(
        "--do_compile",
        action="store_true",
        default=False,
        help="If passes, we will compile the model with `torch.compile`",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="If true, we will used half precision (more or less)`",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=False,
        help="If true, we set torch.backends.cudnn.benchmark = True",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(224, 224),
        help="image size as a tuple",
    )
    args = parser.parse_args()
    model_name, batch_size, do_compile, fp16, image_size = (
        args.model_name,
        args.batch_size,
        args.do_compile,
        args.fp16,
        args.image_size,
    )
    if args.cudnn_benchmark:
        print(f"[INFO] settings torch.backends.cudnn.benchmark = True")
        torch.backends.cudnn.benchmark = True
    pprint(args._get_kwargs())
    model = get_model(model_name)
    run_func = get_run_function(
        model, batch_size, image_size=image_size, do_compile=do_compile, is_fp16=fp16
    )
    mean, median, number_per_run, memory = profile_model(run_func)
    save_to_csv(
        model_name=model_name,
        mean=mean,
        median=median,
        number_per_run=number_per_run,
        memory=memory,
        do_compile=do_compile,
        batch_size=batch_size,
        is_fp16=fp16,
        image_size=image_size[0],  # assuming they are squared
        cudnn_benchmark=args.cudnn_benchmark,
    )
