"""GAN/VAE augmentation with latent-space visualisation (brief item 9).

    python -m surveillance.generative

Question: when one disease is rare, do synthetic leaves help? Setup (64 x 64 PlantVillage photos, the same
8 classes and leaf-grouped splits as fundamentals/):
- "Potato late blight" is cut to 25 training photos; the other 7 classes keep 250 (an imbalanced set).
- A conditional VAE and a conditional DCGAN are trained ONLY on that imbalanced training set.
- A small CNN is trained 5 ways (3 seeds each): real only, + oversampling, + classic augmentation,
  + 225 CVAE late-blight samples, + 225 GAN late-blight samples. Test: 100 photos per class.
Writes surveillance/results/generative_*.png and generative.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "surveillance" / "results"
CACHE = ROOT / "data" / "generative_leaves64.npz"
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from fundamentals.data import CLASSES  # noqa: E402

MINORITY = CLASSES.index("Potato___Late_blight")
MINORITY_N = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHORT = [c.split("___")[0][:6] + " " + c.split("___")[1].replace("_", " ")[:11] for c in CLASSES]


def load_data():
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    sys.path.insert(0, str(ROOT / "Flask Deployed App"))
    from CNN import idx_to_classes
    splits = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    out = {}
    for split, per in (("train", 250), ("test_lab", 100)):
        xs, ys = [], []
        for c, cls in enumerate(CLASSES):
            paths = [p for p, label, dom in splits[split] if dom == "lab" and idx_to_classes[label] == cls]
            n = MINORITY_N if (split == "train" and c == MINORITY) else per
            for p in rng.choice(paths, n, replace=False):
                with Image.open(ROOT / "data" / p) as im:
                    xs.append(np.asarray(im.convert("RGB").resize((64, 64), Image.BILINEAR)))
                ys.append(c)
        out[f"{split}_x"], out[f"{split}_y"] = np.stack(xs), np.array(ys)
    np.savez_compressed(CACHE, **out)
    return out


def to_tensor(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1  # [-1, 1]


def to_image(t):
    return ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()


# ------------------------------------------------------------------ conditional VAE
class CVAE(nn.Module):
    def __init__(self, n_classes=8, latent=32):
        super().__init__()
        self.n_classes, self.latent = n_classes, latent
        self.enc = nn.Sequential(
            nn.Conv2d(3 + n_classes, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), nn.Flatten())
        self.mu = nn.Linear(256 * 16, latent)
        self.logvar = nn.Linear(256 * 16, latent)
        self.fc = nn.Linear(latent + n_classes, 256 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())

    def encode(self, x, y):
        onehot = F.one_hot(y, self.n_classes).float()[:, :, None, None].expand(-1, -1, 64, 64)
        h = self.enc(torch.cat([x, onehot], 1))
        return self.mu(h), self.logvar(h)

    def decode(self, z, y):
        h = self.fc(torch.cat([z, F.one_hot(y, self.n_classes).float()], 1)).view(-1, 256, 4, 4)
        return self.dec(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)  # reparameterisation trick
        return self.decode(z, y), mu, logvar


def train_cvae(x, y, epochs=80, beta=1.0):
    torch.manual_seed(0)
    model = CVAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.utils.data.TensorDataset(to_tensor(x), torch.from_numpy(y).long())
    loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    history = []
    for epoch in range(epochs):
        rec_sum = kl_sum = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            recon, mu, logvar = model(xb, yb)
            rec = F.mse_loss(recon, xb, reduction="sum") / len(xb)
            kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(xb)
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            rec_sum, kl_sum, n = rec_sum + float(rec) * len(xb), kl_sum + float(kl) * len(xb), n + len(xb)
        history.append({"epoch": epoch + 1, "reconstruction": round(rec_sum / n, 2), "kl": round(kl_sum / n, 2)})
    return model.eval(), history


# ------------------------------------------------------------------ conditional DCGAN
class Generator(nn.Module):
    def __init__(self, n_classes=8, z_dim=100):
        super().__init__()
        self.embed = nn.Embedding(n_classes, 50)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + 50, 256, 4, 1, 0), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z, y):
        return self.net(torch.cat([z, self.embed(y)], 1)[:, :, None, None])


class Discriminator(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.embed = nn.Embedding(n_classes, 64 * 64)
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0), nn.Flatten())

    def forward(self, x, y):
        return self.net(torch.cat([x, self.embed(y).view(-1, 1, 64, 64)], 1))


def train_gan(x, y, epochs=150):
    torch.manual_seed(0)
    g, d = Generator().to(DEVICE), Discriminator().to(DEVICE)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    # Balanced sampling so the generator sees the rare class as often as the others.
    w = 1.0 / np.bincount(y)[y]
    sampler = torch.utils.data.WeightedRandomSampler(w, len(y), replacement=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(to_tensor(x), torch.from_numpy(y).long()),
                                         batch_size=64, sampler=sampler)
    history = []
    for epoch in range(epochs):
        ld = lg = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            z = torch.randn(len(xb), 100, device=DEVICE)
            fake = g(z, yb)
            loss_d = bce(d(xb, yb), torch.full((len(xb), 1), 0.9, device=DEVICE)) + \
                bce(d(fake.detach(), yb), torch.zeros(len(xb), 1, device=DEVICE))  # one-sided label smoothing
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            loss_g = bce(d(fake, yb), torch.ones(len(xb), 1, device=DEVICE))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            ld, lg, n = ld + float(loss_d), lg + float(loss_g), n + 1
        history.append({"epoch": epoch + 1, "loss_D": round(ld / n, 3), "loss_G": round(lg / n, 3)})
    return g.eval(), history


# ------------------------------------------------------------------ downstream classifier
class SmallCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.net = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def classic_augment(xb):
    flip = torch.rand(len(xb), device=xb.device) < 0.5
    xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
    vflip = torch.rand(len(xb), device=xb.device) < 0.5
    xb = torch.where(vflip[:, None, None, None], xb.flip(-2), xb)
    return (xb * torch.empty(len(xb), 1, 1, 1, device=xb.device).uniform_(0.85, 1.15)).clamp(-1, 1)


def train_classifier(x, y, seed, augment=False, epochs=25):
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * len(loader))
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if augment:
                xb = classic_augment(xb)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.inference_mode()
def predict(model, x):
    return torch.cat([model(x[i:i + 256].to(DEVICE)).argmax(1).cpu() for i in range(0, len(x), 256)])


def scores(pred, true):
    from sklearn.metrics import f1_score, recall_score
    return {"accuracy": float((pred == true).float().mean()),
            "macro_F1": float(f1_score(true, pred, average="macro")),
            "minority_recall": float(recall_score(true, pred, labels=[MINORITY], average="macro"))}


# ------------------------------------------------------------------ figures
def grid(images, rows, cols, title, name, row_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.15 + 0.4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis("off")
        if row_labels and i % cols == 0:
            ax.set_title(row_labels[i // cols], fontsize=6, loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.savefig(RESULTS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def latent_plots(cvae, x, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    with torch.inference_mode():
        mu = torch.cat([cvae.encode(to_tensor(x[i:i + 256]).to(DEVICE), torch.from_numpy(y[i:i + 256]).long().to(DEVICE))[0].cpu()
                        for i in range(0, len(x), 256)]).numpy()
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (pts, title) in zip(axes, ((PCA(2).fit_transform(mu), "PCA of VAE latent means"),
                                       (TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(mu), "t-SNE of VAE latent means"))):
        for c in range(len(CLASSES)):
            m = y == c
            ax.scatter(pts[m, 0], pts[m, 1], s=7, color=colours[c], label=SHORT[c], alpha=0.75, edgecolors="none")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(fontsize=7, markerscale=2, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("Latent-space organisation of test leaves (the CVAE is conditioned on the class, so the latent\n"
                 "codes describe appearance within a class: lighting, leaf shape, background)", x=0.01, ha="left", fontsize=9)
    fig.savefig(RESULTS / "generative_latent_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Interpolation between two test leaves of the same class, decoded through the latent space.
    a, b = 0, int(np.where(y == y[0])[0][5])
    za, zb = mu[a], mu[b]
    zs = torch.from_numpy(np.stack([za + t * (zb - za) for t in np.linspace(0, 1, 8)])).float().to(DEVICE)
    with torch.inference_mode():
        imgs = to_image(cvae.decode(zs, torch.full((8,), int(y[a]), device=DEVICE)))
    grid(imgs, 1, 8, "Latent interpolation between two leaves (CVAE decoder)", "generative_interpolation")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = load_data()
    x, y, xt, yt = d["train_x"], d["train_y"], d["test_lab_x"], d["test_lab_y"]
    out = {"setup": {"classes": CLASSES, "minority_class": CLASSES[MINORITY], "minority_train_images": MINORITY_N,
                     "other_train_images_per_class": 250, "test_images_per_class": 100, "image_size": 64}}
    t0 = time.time()
    cvae, cvae_hist = train_cvae(x, y)
    g, gan_hist = train_gan(x, y)
    out["training"] = {"cvae_last": cvae_hist[-1], "gan_last": gan_hist[-1], "minutes": round((time.time() - t0) / 60, 1)}

    n_syn = 250 - MINORITY_N
    with torch.inference_mode():
        lab = torch.full((n_syn,), MINORITY, device=DEVICE)
        syn_vae = cvae.decode(torch.randn(n_syn, cvae.latent, device=DEVICE), lab).cpu()
        syn_gan = g(torch.randn(n_syn, 100, device=DEVICE), lab).cpu()
        per = torch.arange(len(CLASSES), device=DEVICE).repeat_interleave(8)
        grid(to_image(cvae.decode(torch.randn(len(per), cvae.latent, device=DEVICE), per)), len(CLASSES), 8,
             "Conditional VAE samples (one row per class)", "generative_cvae_samples", SHORT)
        grid(to_image(g(torch.randn(len(per), 100, device=DEVICE), per)), len(CLASSES), 8,
             "Conditional GAN samples (one row per class)", "generative_gan_samples", SHORT)
    grid(np.concatenate([x[y == MINORITY][:8], to_image(syn_vae[:8]), to_image(syn_gan[:8])]), 3, 8,
         "Potato late blight: real (top), CVAE (middle), GAN (bottom)", "generative_minority_comparison",
         ["real", "CVAE", "GAN"])
    latent_plots(cvae, xt, yt)

    xr, yr = to_tensor(x), torch.from_numpy(y).long()
    xte, yte = to_tensor(xt), torch.from_numpy(yt).long()
    mino = torch.where(yr == MINORITY)[0]
    over = mino[torch.randint(len(mino), (n_syn,), generator=torch.Generator().manual_seed(0))]
    variants = {
        "real only (imbalanced)": (xr, yr, False),
        "+ oversampling (repeat the 25)": (torch.cat([xr, xr[over]]), torch.cat([yr, yr[over]]), False),
        "+ classic augmentation": (xr, yr, True),
        "+ 225 CVAE samples": (torch.cat([xr, syn_vae]), torch.cat([yr, torch.full((n_syn,), MINORITY)]), False),
        "+ 225 GAN samples": (torch.cat([xr, syn_gan]), torch.cat([yr, torch.full((n_syn,), MINORITY)]), False),
    }
    results = {}
    real_model = None
    for name, (xa, ya, aug) in variants.items():
        runs = []
        for seed in (0, 1, 2):
            m = train_classifier(xa, ya, seed, augment=aug)
            runs.append(scores(predict(m, xte), yte))
            if name.startswith("real only") and seed == 0:
                real_model = m
        results[name] = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                             "std": round(float(np.std([r[k] for r in runs])), 4)} for k in runs[0]}
        print(f"{name:34s} macro-F1 {results[name]['macro_F1']['mean']:.3f}  "
              f"minority recall {results[name]['minority_recall']['mean']:.3f} +/- {results[name]['minority_recall']['std']:.3f}")
    out["augmentation_study"] = results
    # How often a classifier trained on real data recognises the synthetic leaves as late blight.
    out["synthetic_recognised_as_minority"] = {
        "CVAE": round(float((predict(real_model, syn_vae) == MINORITY).float().mean()), 4),
        "GAN": round(float((predict(real_model, syn_gan) == MINORITY).float().mean()), 4)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names = list(results)
    vals = [results[n]["minority_recall"]["mean"] * 100 for n in names]
    errs = [results[n]["minority_recall"]["std"] * 100 for n in names]
    ax.barh(names, vals, xerr=errs, color="#2a78d6", height=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 2, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("recall on potato late blight (rare class), test set, mean of 3 seeds")
    ax.set_title("Does synthetic data help a rare disease?", loc="left")
    ax.grid(axis="x", color="#e1e0d9")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "generative_augmentation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot([h["reconstruction"] for h in cvae_hist], color="#2a78d6", label="reconstruction")
    axes[0].plot([h["kl"] for h in cvae_hist], color="#eb6834", label="KL")
    axes[0].set_title("CVAE loss terms", loc="left")
    axes[0].legend(frameon=False)
    axes[1].plot([h["loss_D"] for h in gan_hist], color="#2a78d6", label="discriminator")
    axes[1].plot([h["loss_G"] for h in gan_hist], color="#eb6834", label="generator")
    axes[1].set_title("GAN losses", loc="left")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "generative_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    torch.save({"cvae": cvae.state_dict(), "gan_generator": g.state_dict()}, ROOT / "surveillance" / "runs" / "generative.pt")
    (RESULTS / "generative.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["synthetic_recognised_as_minority"]))


if __name__ == "__main__":
    main()
