"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network with learnable gate parameters that
encourage sparse (pruned) weight matrices during training via L1 regularization.

Author : (Your Name)
Date   : April 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – safe for any environment
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. PrunableLinear Layer
# ──────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns whether each individual
    weight should be kept or removed.

    For every weight w_ij we introduce a learnable scalar gate_score g_ij.
    The gate is obtained via sigmoid(g_ij), squashing the raw score to (0, 1).
    The effective weight used in the forward pass is  w_ij * sigmoid(g_ij).

    Because sigmoid is smooth everywhere, gradients flow back through both
    `weight` and `gate_scores` via standard autograd – no custom backward
    needed.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same as nn.Linear)
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Gate scores – same shape as weight, initialised near 1 so the
        # network starts with most gates open and learns to close them.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        # Kaiming uniform for weights (standard for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        # Initialise gate_scores so sigmoid(g) ≈ 0.9  →  g ≈ 2.2
        nn.init.constant_(self.gate_scores, 2.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- gate computation ---
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in) ∈ (0,1)

        # --- pruned effective weights ---
        pruned_weights = self.weight * gates             # element-wise product

        # --- linear transform (implemented from scratch via matmul + bias) ---
        # x : (..., in_features)
        # pruned_weights.T : (in_features, out_features)
        return x @ pruned_weights.t() + self.bias

    # ── helpers ──────────────────────────────────────────────────────────────

    def current_gates(self) -> torch.Tensor:
        """Return detached gate values (no grad) for inspection."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold`."""
        g = self.current_gates()
        return (g < threshold).float().mean().item()


# ──────────────────────────────────────────────
# 2. Network Definition
# ──────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A simple 3-hidden-layer feed-forward network.
    CIFAR-10 images (3×32×32) are flattened to 3072-d vectors.
    """

    def __init__(self, hidden: int = 512, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            PrunableLinear(3072, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),

            PrunableLinear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(self.flatten(x))

    def prunable_layers(self):
        """Yield every PrunableLinear in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ──────────────────────────────────────────────
# 3. Sparsity Loss
# ──────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of all gate values = sum of sigmoid(gate_scores) across all
    PrunableLinear layers.  Minimising this encourages gates → 0.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        total = total + torch.sigmoid(layer.gate_scores).sum()
    return total


# ──────────────────────────────────────────────
# 4. Data Loading
# ──────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────
# 5. Training & Evaluation
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_loss = total_cls = total_sp = 0.0
    correct = 0
    n = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = sparsity_loss(model)
        loss     = cls_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()  * images.size(0)
        total_cls  += cls_loss.item() * images.size(0)
        total_sp   += sp_loss.item()  * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += images.size(0)

    return {
        "loss"    : total_loss / n,
        "cls_loss": total_cls  / n,
        "sp_loss" : total_sp   / n,
        "acc"     : correct / n * 100,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        n       += images.size(0)
    return correct / n * 100


def overall_sparsity(model: SelfPruningNet, threshold: float = 1e-2) -> float:
    """Fraction of all gates (across every PrunableLinear) below threshold."""
    below = total = 0
    for layer in model.prunable_layers():
        g = layer.current_gates()
        below += (g < threshold).sum().item()
        total += g.numel()
    return below / total * 100


# ──────────────────────────────────────────────
# 6. Experiment Runner
# ──────────────────────────────────────────────

def run_experiment(lam: float, epochs: int, device, train_loader, test_loader,
                   hidden: int = 512) -> dict:
    print(f"\n{'='*55}")
    print(f"  λ = {lam:g}   |   epochs = {epochs}")
    print(f"{'='*55}")

    model = SelfPruningNet(hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            sp = overall_sparsity(model)
            print(f"  Ep {epoch:3d}/{epochs}  "
                  f"loss={stats['loss']:.4f}  "
                  f"cls={stats['cls_loss']:.4f}  "
                  f"sparse={sp:.1f}%  "
                  f"train_acc={stats['acc']:.1f}%")

    test_acc  = evaluate(model, test_loader, device)
    sparsity  = overall_sparsity(model)

    print(f"\n  ► Test Accuracy : {test_acc:.2f}%")
    print(f"  ► Sparsity      : {sparsity:.2f}%")

    # Collect all gate values for plotting
    all_gates = torch.cat([
        layer.current_gates().flatten()
        for layer in model.prunable_layers()
    ]).cpu().numpy()

    return {
        "lam"      : lam,
        "test_acc" : test_acc,
        "sparsity" : sparsity,
        "gates"    : all_gates,
        "model"    : model,
    }


# ──────────────────────────────────────────────
# 7. Plotting
# ──────────────────────────────────────────────

def plot_gate_distributions(results: list, save_path: str = "gate_distributions.png"):
    """
    For each experiment, plot a histogram of final gate values.
    A successful run shows a large spike near 0 and a second cluster near 1.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#E63946", "#2A9D8F", "#E9C46A"]

    for ax, res, color in zip(axes, results, colors):
        gates = res["gates"]
        ax.hist(gates, bins=80, range=(0, 1), color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"λ = {res['lam']:g}\n"
            f"Acc={res['test_acc']:.1f}%  Sparse={res['sparsity']:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Gate value (sigmoid output)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Gate Value Distributions After Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Plot saved → {save_path}]")


# ──────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=256)

    # Three λ values: low / medium / high
    lambda_values = [1e-5, 1e-4, 5e-4]
    epochs        = 30          # increase to 60+ for better accuracy

    results = []
    for lam in lambda_values:
        res = run_experiment(lam, epochs, device, train_loader, test_loader)
        results.append(res)

    # ── Summary Table ──────────────────────────────────────────────────────
    print("\n\n" + "="*50)
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("="*50)
    for r in results:
        print(f"  {r['lam']:<12g} {r['test_acc']:>14.2f} {r['sparsity']:>14.2f}")
    print("="*50)

    # ── Gate distribution plots ────────────────────────────────────────────
    plot_gate_distributions(results, save_path="gate_distributions.png")

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
