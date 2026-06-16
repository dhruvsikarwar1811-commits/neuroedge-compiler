import numpy as np
import matplotlib.pyplot as plt


def analyze_sparsity(weights, thresholds=None):
    """Analyze what fraction of weights fall below various magnitude thresholds."""
    if thresholds is None:
        thresholds = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    total = weights.size
    abs_weights = np.abs(weights)

    results = []
    for t in thresholds:
        sparse_count = np.sum(abs_weights <= t)
        sparsity_pct = 100.0 * sparse_count / total
        results.append((t, sparsity_pct))

    return results


def layer_sparsity_report(weights):
    """Print a detailed sparsity report for a weight tensor."""
    print("=" * 45)
    print(" NeuroEdge — Sparsity Analysis")
    print("=" * 45)
    print(f"Tensor shape: {weights.shape}")
    print(f"Total params: {weights.size}")
    print(f"Mean |weight|: {np.mean(np.abs(weights)):.6f}")
    print(f"Max |weight|:  {np.max(np.abs(weights)):.6f}")
    print("-" * 45)

    results = analyze_sparsity(weights)
    for t, pct in results:
        print(f"Threshold {t:.2f}: {pct:6.2f}% of weights")

    return results


def plot_sparsity(weights, results):
    thresholds = [r[0] for r in results]
    percentages = [r[1] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar([str(t) for t in thresholds], percentages, color='#10b981')
    axes[0].set_title('Sparsity vs Threshold')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('% weights below threshold')

    axes[1].hist(weights.flatten(), bins=60, color='#7c3aed')
    axes[1].set_title('Weight Distribution')
    axes[1].set_xlabel('Weight value')
    axes[1].set_ylabel('Count')

    plt.suptitle('NeuroEdge — Sparsity Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/sparsity_analysis.png', dpi=150)
    print("\nPlot saved: results/sparsity_analysis.png")


if __name__ == '__main__':
    np.random.seed(42)
    weights = np.random.randn(128, 128).astype(np.float32) * 0.1
    results = layer_sparsity_report(weights)
    plot_sparsity(weights, results)
