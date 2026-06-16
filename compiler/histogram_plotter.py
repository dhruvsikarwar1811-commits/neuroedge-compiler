import numpy as np
import matplotlib.pyplot as plt


def plot_weight_histogram(layers: dict, bins=80):
    """Plot per-layer and combined weight histograms."""
    n = len(layers)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    all_weights = []

    for i, (name, w) in enumerate(layers.items()):
        flat = w.flatten()
        all_weights.append(flat)

        # Per-layer histogram
        axes[0, i].hist(flat, bins=bins, color='#6366f1', alpha=0.85)
        axes[0, i].set_title(f'{name}')
        axes[0, i].set_xlabel('Weight value')
        axes[0, i].set_ylabel('Count')

        mean = np.mean(flat)
        std  = np.std(flat)
        axes[0, i].axvline(mean, color='#f59e0b', linewidth=1.5, label=f'μ={mean:.3f}')
        axes[0, i].axvline(mean + std, color='#ef4444', linewidth=1, linestyle='--', label=f'σ={std:.3f}')
        axes[0, i].axvline(mean - std, color='#ef4444', linewidth=1, linestyle='--')
        axes[0, i].legend(fontsize=7)

        # Cumulative distribution
        sorted_w = np.sort(np.abs(flat))
        cdf = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
        axes[1, i].plot(sorted_w, cdf, color='#10b981', linewidth=1.5)
        axes[1, i].set_title(f'{name} — CDF')
        axes[1, i].set_xlabel('|Weight|')
        axes[1, i].set_ylabel('Cumulative fraction')
        axes[1, i].axvline(0.05, color='#f59e0b', linestyle='--', label='t=0.05')
        axes[1, i].legend(fontsize=7)

    plt.suptitle('NeuroEdge — Weight Histograms & CDF', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/weight_histograms.png', dpi=150)
    print("Plot saved: results/weight_histograms.png")


def print_histogram_stats(layers: dict):
    print("=" * 55)
    print(" NeuroEdge — Weight Distribution Stats")
    print("=" * 55)
    print(f"{'Layer':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'<0.05%':>8}")
    print("-" * 55)
    for name, w in layers.items():
        flat = w.flatten()
        pct  = 100.0 * np.sum(np.abs(flat) < 0.05) / flat.size
        print(f"{name:<12} {np.mean(flat):>8.4f} {np.std(flat):>8.4f} "
              f"{np.min(flat):>8.4f} {np.max(flat):>8.4f} {pct:>7.1f}%")
    print("=" * 55)


if __name__ == '__main__':
    np.random.seed(42)
    layers = {
        'conv1':  np.random.randn(32, 1, 3, 3).astype(np.float32) * 0.1,
        'conv2':  np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.1,
        'fc1':    np.random.randn(128, 256).astype(np.float32) * 0.05,
        'fc2':    np.random.randn(10, 128).astype(np.float32) * 0.1,
    }
    print_histogram_stats(layers)
    plot_weight_histogram(layers)
