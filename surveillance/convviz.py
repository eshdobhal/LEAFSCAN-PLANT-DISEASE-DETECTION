"""Visualising 2D and 3D convolutions (brief item 7).

    python -m surveillance.convviz

1. A 2D convolution computed by hand (explicit loops) on a real leaf with classic kernels.
2. Learned first-layer filters and feature maps of the trained ResNet-50 baseline.
3. Grad-CAM of the shipped CAFormer-S18: which part of the leaf drove the diagnosis.
4. A 3D convolution over a lesion time-lapse (frames x height x width): a temporal-difference kernel
   lights up tissue that became damaged between frames, which is what progression monitoring needs.
Writes surveillance/results/conv_*.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "surveillance" / "results"
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "Flask Deployed App"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def conv2d_by_hand(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'Valid' 2D cross-correlation (what deep-learning libraries call convolution), written as loops:
    out[i, j] = sum_{u, v} image[i + u, j + v] * kernel[u, v]."""
    kh, kw = kernel.shape
    out = np.zeros((image.shape[0] - kh + 1, image.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


def conv3d_by_hand(volume: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """The same idea with a third (time) axis: out[t, i, j] = sum_{s,u,v} vol[t+s, i+u, j+v] * k[s, u, v]."""
    kt, kh, kw = kernel.shape
    T, H, W = volume.shape
    out = np.zeros((T - kt + 1, H - kh + 1, W - kw + 1))
    for t in range(out.shape[0]):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                out[t, i, j] = np.sum(volume[t:t + kt, i:i + kh, j:j + kw] * kernel)
    return out


def leaf_photo(name="potato_late_blight.JPG", size=128):
    with Image.open(ROOT / "test_images" / name) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float64) / 255


def fig_conv2d(out):
    rgb = leaf_photo()
    grey = rgb.mean(2)
    kernels = {
        "Sobel x (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
        "Sobel y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
        "Laplacian (spots, blobs)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
        "5x5 mean (blur)": np.ones((5, 5)) / 25,
    }
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("leaf (potato late blight)", fontsize=9, loc="left")
    axes[1, 0].imshow(grey, cmap="gray")
    axes[1, 0].set_title("grey input, 128 x 128", fontsize=9, loc="left")
    check = {}
    for col, (name, k) in enumerate(kernels.items(), start=1):
        res = conv2d_by_hand(grey, k)
        lib = ndimage.correlate(grey, k, mode="constant")[k.shape[0] // 2:-(k.shape[0] // 2), k.shape[1] // 2:-(k.shape[1] // 2)]
        check[name] = float(np.abs(res - lib).max())
        axes[0, col].imshow(k, cmap="RdBu_r", vmin=-np.abs(k).max(), vmax=np.abs(k).max())
        for (i, j), v in np.ndenumerate(k):
            if k.shape[0] <= 3:
                axes[0, col].text(j, i, f"{v:g}", ha="center", va="center", fontsize=9)
        axes[0, col].set_title(f"kernel: {name}", fontsize=8, loc="left")
        axes[1, col].imshow(np.abs(res) if "Sobel" in name or "Lap" in name else res, cmap="magma")
        axes[1, col].set_title(f"output {res.shape[0]} x {res.shape[1]}", fontsize=9, loc="left")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("2D convolution by hand: each output pixel is an inner product of a 3x3 (or 5x5) window with the kernel",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(RESULTS / "conv_2d_by_hand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    out["conv2d_max_difference_vs_scipy"] = check


def fig_resnet(out):
    from common import RUNS, build_model, run_arch
    trained = (RUNS / "resnet50" / "best.pt").exists()
    net = build_model(pretrained=not trained, arch="resnet50")
    if trained:
        state = torch.load(RUNS / "resnet50" / "best.pt", map_location="cpu", weights_only=True)
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    net.eval()
    out["resnet_weights"] = "trained LeafScan ResNet-50 baseline" if trained else "ImageNet ResNet-50"
    w = net.conv1.weight.detach().numpy()  # 64 filters, 3 x 7 x 7
    fig, axes = plt.subplots(4, 16, figsize=(14, 3.8))
    for ax, f in zip(axes.flat, w):
        f = f.transpose(1, 2, 0)
        ax.imshow((f - f.min()) / (f.max() - f.min() + 1e-9))
        ax.axis("off")
    fig.suptitle(f"The 64 learned first-layer filters (7x7x3) of the {out['resnet_weights']}: edge, colour and spot detectors",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(RESULTS / "conv_resnet_filters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Feature maps at increasing depth.
    from torchvision.transforms import functional as F
    x = F.normalize(torch.from_numpy(leaf_photo(size=224)).permute(2, 0, 1).float(),
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])[None]
    maps = {}
    with torch.no_grad():
        h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        maps["conv1 (112->56 px)"] = h
        for name in ("layer1", "layer2", "layer3", "layer4"):
            h = getattr(net, name)(h)
            maps[f"{name} ({h.shape[-1]} px, {h.shape[1]} ch)"] = h
    fig, axes = plt.subplots(len(maps), 8, figsize=(12, 7.5))
    for row, (name, m) in enumerate(maps.items()):
        strongest = m[0].mean((1, 2)).argsort(descending=True)[:8]
        for col, c in enumerate(strongest):
            axes[row, col].imshow(m[0, c].numpy(), cmap="viridis")
            axes[row, col].axis("off")
        axes[row, 0].set_title(name, fontsize=8, loc="left")
    fig.suptitle("ResNet-50 feature maps of one leaf, deeper = coarser and more abstract (8 most active channels)",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(RESULTS / "conv_resnet_feature_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def grad_cam(net, x, layer, class_idx=None):
    feats, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("v", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    logits = net(x)
    c = int(logits.argmax(1)) if class_idx is None else class_idx
    net.zero_grad()
    logits[0, c].backward()
    h1.remove()
    h2.remove()
    weights = grads["v"].mean((2, 3), keepdim=True)  # how much each channel matters for this class
    cam = torch.relu((weights * feats["v"]).sum(1))[0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam.detach().numpy(), c, float(logits.softmax(1)[0, c])


def fig_gradcam(out):
    from model import Predictor, make_transform  # the Flask app's own loader
    from CNN import idx_to_classes
    pred = Predictor([(str(ROOT / "Flask Deployed App" / "leafscan_hybrid.pt"), 224)], kind="hybrid")
    net = pred.members[0][0]
    tf = make_transform("hybrid", 224)
    names = ["potato_late_blight.JPG", "tomato_bacterial_spot.JPG", "corn_common_rust.JPG", "apple_black_rot.JPG"]
    fig, axes = plt.subplots(2, len(names), figsize=(12, 6.2))
    for col, name in enumerate(names):
        with Image.open(ROOT / "test_images" / name) as im:
            im = im.convert("RGB")
            x = tf(im)[None].requires_grad_(True)
            disp = np.asarray(im.resize((256, 256)).crop((16, 16, 240, 240)))
        cam, c, p = grad_cam(net, x, net.stages[3])
        cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255
        axes[0, col].imshow(disp)
        axes[0, col].set_title(name.split(".")[0].replace("_", " "), fontsize=8, loc="left")
        axes[1, col].imshow(disp)
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        label = idx_to_classes[c].replace("___", " · ").replace("_", " ")
        axes[1, col].set_title(f"{label[:34]} ({p:.0%})", fontsize=8, loc="left")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Grad-CAM on the shipped CAFormer-S18 (last attention stage): warm = evidence for the predicted class",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(RESULTS / "conv_gradcam.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    out["gradcam_finding"] = (
        "For late blight the evidence sits on the lesion. For bacterial spot, common rust and black rot much of it "
        "sits on the background and image borders: PlantVillage photographed each class against similar backdrops, "
        "so the model partly learned the backdrop. This 'shortcut' is one reason field accuracy (71%) is far below "
        "lab accuracy (99.6%); see the error analysis and model card.")


def fig_conv3d(out):
    """Build a time-lapse from a real lesion mask: the lesion grows by dilation, frame by frame."""
    data = ROOT / "data" / "leafseg"
    test = json.loads((data / "splits.json").read_text())["test"]
    # The test photo with the most damaged tissue, so the growth is visible at 96 x 96.
    name = max(test, key=lambda n: (np.asarray(Image.open(data / "masks" / f"{n}.png")) > 127).mean())
    rgb = np.asarray(Image.open(data / "images" / f"{name}.jpg").convert("RGB").resize((96, 96)), dtype=np.float64) / 255
    lesion = np.asarray(Image.open(data / "masks" / f"{name}.png").resize((96, 96), Image.NEAREST)) > 127
    # Grow each lesion from its core (far from its edge) outwards, reaching the true mask on the last frame.
    depth = ndimage.distance_transform_edt(lesion)
    frames, masks = [], []
    for t in range(8):  # day 0..7: the lesion spreads outward and darkens the tissue it reaches
        m = lesion & (depth >= depth.max() * (1 - (t + 1) / 8))
        f = rgb.copy()
        f[m] = f[m] * 0.45 + np.array([0.25, 0.18, 0.08]) * 0.55  # brown, necrotic tissue
        frames.append(f.mean(2))
        masks.append(m)
    vol = np.stack(frames)  # (time, height, width)
    kernel = np.zeros((2, 3, 3))
    kernel[0] = -1 / 9
    kernel[1] = 1 / 9  # mean brightness at t+1 minus mean at t, over a 3x3 patch
    change = -conv3d_by_hand(vol, kernel)  # positive where tissue darkened, i.e. newly damaged
    if change.max() < 1e-3:
        raise RuntimeError("the time-lapse frames do not change; pick a photo with visible lesions")
    torch_check = torch.nn.functional.conv3d(torch.from_numpy(vol)[None, None], torch.from_numpy(kernel)[None, None])[0, 0].numpy()
    out["conv3d_max_difference_vs_torch"] = float(np.abs(-change - torch_check).max())
    fig, axes = plt.subplots(3, 7, figsize=(14, 6.3))
    for t in range(7):
        axes[0, t].imshow(frames[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"frame t={t}", fontsize=8, loc="left")
        axes[1, t].imshow(change[t], cmap="magma", vmin=0, vmax=change.max())
        axes[1, t].set_title(f"3D conv: t={t}->{t + 1}", fontsize=8, loc="left")
        axes[2, t].imshow(masks[t + 1] & ~masks[t], cmap="Greens")
        axes[2, t].set_title("true new damage", fontsize=8, loc="left")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("3D convolution (2 x 3 x 3 kernel over time x height x width) on a lesion time-lapse built from a real "
                 "mask: it responds to\nnew damage only, not to tissue that was already damaged; a 2D kernel on one frame cannot see change",
                 x=0.01, ha="left", fontsize=9)
    fig.savefig(RESULTS / "conv_3d_timelapse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pred_new = change > change.max() * 0.25
    true_new = np.stack([masks[t + 1] & ~masks[t] for t in range(7)])[:, 1:-1, 1:-1]
    out["conv3d_new_damage_IoU"] = round(float((pred_new & true_new).sum() / max(1, (pred_new | true_new).sum())), 3)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {}
    fig_conv2d(out)
    fig_resnet(out)
    fig_gradcam(out)
    fig_conv3d(out)
    (RESULTS / "convviz.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
