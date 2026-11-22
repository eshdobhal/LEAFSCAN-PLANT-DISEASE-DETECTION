"""Convert the two trained CAFormer-S18 models to ONNX for the browser version (web/).

    python web/export_onnx.py OUT_DIR [--check N]

Needs torch, timm, onnx and onnxruntime. For each model it:
1. exports the PyTorch network to ONNX (batch size dynamic, so the photo and its mirror run together);
2. stores the weights in half precision (49 MB per model, like the .pt files) with a Cast to float32 in
   front of every weight, so the browser computes in float32 exactly like the server does;
3. with --check N, compares ONNX Runtime with PyTorch on N field test photos (needs data/).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "Flask Deployed App"
sys.path.insert(0, str(APP))
from CNN import idx_to_classes  # noqa: E402  (the 39 classes in the app's order)

MODELS = {"leafscan_224": ("leafscan_hybrid.pt", 224), "leafscan_288": ("leafscan_hybrid_288.pt", 288)}


def load_network(weights: Path) -> torch.nn.Module:
    import timm
    from timm.layers import set_fused_attn
    set_fused_attn(False)  # plain matmul + softmax attention: supported by every ONNX Runtime Web backend
    net = timm.create_model("caformer_s18", pretrained=False, num_classes=len(idx_to_classes))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return net.eval()


def weights_to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Store large float32 weights as float16 and add a Cast back to float32 (halves the file size)."""
    graph = model.graph
    casts, converted = [], []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < 64:
            continue  # small tensors and integer shape constants stay as they are
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__cast"))
        converted.append(half)
        graph.initializer.remove(init)
    graph.initializer.extend(converted)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def export(name: str, out_dir: Path) -> Path:
    weights, size = MODELS[name]
    net = load_network(APP / weights)
    tmp = out_dir / f"{name}.fp32.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, size, size), tmp, dynamo=False, opset_version=17,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    model = weights_to_fp16(onnx.load(tmp))
    onnx.checker.check_model(model)
    path = out_dir / f"{name}.onnx"
    onnx.save(model, path)
    tmp.unlink()
    print(f"{path.name}: {path.stat().st_size / 2**20:.1f} MB")
    return path


def check(out_dir: Path, n: int):
    """ONNX Runtime vs PyTorch on real photos, using the app's own preprocessing."""
    import json

    import onnxruntime as ort
    from PIL import Image

    from model import make_transform
    items = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))["test_field"][:n]
    for name, (weights, size) in MODELS.items():
        net = load_network(APP / weights)
        session = ort.InferenceSession(str(out_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        tf = make_transform("hybrid", size)
        worst, agree = 0.0, 0
        for path, _, _ in items:
            x = tf(Image.open(ROOT / "data" / path).convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                p_torch = net(x).softmax(1).numpy()
            logits = session.run(None, {"image": x.numpy()})[0]
            p_onnx = np.exp(logits - logits.max(1, keepdims=True))
            p_onnx /= p_onnx.sum(1, keepdims=True)
            worst = max(worst, float(np.abs(p_torch - p_onnx).max()))
            agree += int(p_torch.argmax() == p_onnx.argmax())
        print(f"{name}: top-1 agreement {agree}/{len(items)}, largest probability difference {worst:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--check", type=int, default=0, help="compare with PyTorch on N field test photos")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_name in MODELS:
        export(model_name, args.out_dir)
    if args.check:
        check(args.out_dir, args.check)
