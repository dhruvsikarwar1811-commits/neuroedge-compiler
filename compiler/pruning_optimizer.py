import numpy as np
import matplotlib.pyplot as plt


def evaluate_pruning(weights, threshold):
    """Prune weights below threshold and compute stats."""
    mask = np.abs(weights) > threshold
    pruned = weights * mask
    sparsity = 100.0 * (1 - np.sum(mask) / weights.size)
    error = np.mean(np.abs(weights - pruned))
    return pruned, sparsity, error


def find_optimal_threshold(weights, target_sparsity=70.0):
    """Binary search for threshold that hits target sparsity."""
    lo, hi = 0.0, np.max(np.abs(weights))
    for _ in range(50):
        mid = (lo + hi) / 2
        _, sparsity, _ = evaluate_pruning(weights, mid)
        if sparsity < target_sparsity:
            lo = mid
        else:
            hi = mid
    _, sparsity, error = evaluate_pruning(weights, mid)
    return mid, sparsity, error


def threshold_sweep(weights, n=20):
    """Sweep thresholds and return sparsity/error curve."""
    thresholds = np.linspace(0, np.max(np.abs(weights)) * 0.9, n)
    results = []
    for t in thresholds:
        _, sparsity, error = evaluate_pruning(weights, t)
        results.append((t, sparsity, error))
    return results


def print_pruning_report(layers: dict, target_sparsity=70.0):
    print("=" * 60)
    print(" NeuroEdge — Pruning Threshold Optimizer")
    print("=" * 60)
    print(f"{'Layer':<12} {'Opt Threshold':>14} {'Sparsity':>10} {'Error':>10}")
    print("-" * 60)

    report = []
    for name, w in layers.items():
        t_opt, sparsity, error = find_optimal_threshold(w, target_sparsity)
        print(f"{name:<12} {t_opt:>14.6f} {sparsity:>9.1f}% {error:>10.6f}")
        report.append({'name': name, 'threshold': t_opt,
                       'sparsity': sparsity, 'error': error})
    print("=" * 60)
    return report


def plot_pruning_curves(weights, layer_name='fc1'):
    results = threshold_sweep(weights)
    thresholds = [r[0] for r in results]
    sparsities = [r[1] for r in results]
    errors     = [r[2] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(thresholds, sparsities, color='#10b981', linewidth=2, marker='o', markersize=4)
    axes[0].set_title(f'{layer_name} — Sparsity vs Threshold')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Sparsity %')
    axes[0].axhline(70, color='#f59e0b', linestyle='--', label='Target 70%')
    axes[0].legend()

    axes[1].plot(thresholds, errors, color='#ef4444', linewidth=2, marker='o', markersize=4)
    axes[1].set_title(f'{layer_name} — Error vs Threshold')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Mean Abs Error')

    plt.suptitle('NeuroEdge — Pruning Threshold Optimizer', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/pruning_optimizer.png', dpi=150)
    print("\nPlot saved: results/pruning_optimizer.png")


if __name__ == '__main__':
    np.random.seed(42)
    layers = {
        'conv1':  np.random.randn(32, 1, 3, 3).astype(np.float32) * 0.1,
        'conv2':  np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.1,
        'fc1':    np.random.randn(128, 256).astype(np.float32) * 0.05,
        'fc2':    np.random.randn(10, 128).astype(np.float32) * 0.1,
    }
    report = print_pruning_report(layers, target_sparsity=70.0)
    plot_pruning_curves(layers['fc1'], layer_name='fc1')
