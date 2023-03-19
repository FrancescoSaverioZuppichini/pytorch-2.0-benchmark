#!/bin/bash
# thanks daddyGTP

NAMES=("resnet34" "resnext50_32x4d" "clip_vision_vit-b/32" "clip_vision_vit-l/14" "clip_vision_vit-rn50")

# List of base sizes to iterate over
BATCH_SIZES=(1 4 8 16 32 64)

# Loop over the names
for name in "${NAMES[@]}"
do
    # Loop over the base sizes
    for size in "${BATCH_SIZES[@]}"
    do
        echo "Running script for $name with batch size $size"
        # python main.py "$name" --batch_size "$size" --fp16 
        # python main.py "$name" --batch_size "$size" --fp16 --do_compile
        # python main.py "$name" --batch_size "$size" --fp16 --cudnn_benchmark

        # clip vit needs a 224x224 image
        if [ "$name" = "resnet34" ] || [ "$name" = "resnext50_32x4d" ]; then
            echo "Running with bigger image"
            # python main.py "$name" --batch_size "$size" --fp16 --image-size 640 640
            # python main.py "$name" --batch_size "$size" --fp16 --do_compile --image-size 640 640
            python main.py "$name" --batch_size "$size" --fp16 --image-size 640 640  --cudnn_benchmark

        fi

    done
done