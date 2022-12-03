import clip
import onnxruntime as ort
import torch

MODEL_NAME = "ViT-B/32"
model_name = "ViT-B-32.onnx"
model, preprocess = clip.load(MODEL_NAME, jit=False, device="cuda")

with torch.autocast("cuda", dtype=torch.float16):
    image_encoder = model.visual.eval()
    x = torch.randn(1, 3, 224, 224, device="cuda")
    # # Export the model
    torch.onnx.export(
        image_encoder,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        model_name,  # where to save the model (can be a file or file-like object)
        opset_version=16,
        export_params=True,  # store the trained parameter weights inside the model file
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["image"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "image": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

# let's check
print("Checking")
x = torch.randn(1, 3, 224, 224, device="cuda")
ort_session = ort.InferenceSession(model_name, providers=["CUDAExecutionProvider"])
outputs = ort_session.run(None, {"image": x.cpu().numpy()})
print(outputs[0].shape, outputs[0].dtype)
