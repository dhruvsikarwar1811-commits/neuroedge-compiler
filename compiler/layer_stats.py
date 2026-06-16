import numpy as np
import matplotlib.pyplot as plt


def compute_layer_stats(layers: dict):
    """
    Compute compression stats for each layer.
    layers: dict of {layer_name: weight_array}
    """
    print("=" * 60)
    print(" NeuroEdge — Layer-wise Compression Stats")
    print("=" * 60)
    print(f"{'Layer':<20} {'Shape':<15} {'Params':>8} {'Sparsity':>10} {'CR INT8':>9} {'CR INT4':>9}")
    print("-" * 60)

    report = []
    for name, w in layers.items():
        params = w.size
        sparsity = 100.0 * np.sum(np.abs(w) < 0.05) / params
        cr_int8 = 4.0   # FP32 -> INT8 = 4x
        cr_int4 = 8.0   # FP32 -> INT4 = 8x
        shape_str = str(w.shape)
        print(f"{name:<20} {shape_str:<15} {params:>8} {sparsity:>9.1f}% {cr_int8:>8.1f}x {cr_int4:>8.1f}x")
        report.append({
            'name': name, 'params': params,
            'sparsity': sparsity, 'cr_int8': cr_int8, 'cr_int4': cr_int4
        })

    total = sum(r['params'] for r in report)
    avg_sp = np.mean([r['sparsity'] for r in report])
    print("-" * 60)
    print(f"{'TOTAL':<20} {'':<15} {total:>8} {avg_sp:>9.1f}%")
    return report


def plot_layer_stats(report):
    names = [r['name'] for r in report]
    params = [r['params'] for r in report]
    sparsity = [r['sparsity'] for r in report]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].barh(names, params, color='#3b82f6')
    axes[0].set_title('Parameters per Layer')
    axes[0].set_xlabel('Parameter Count')

    axes[1].barh(names, sparsity, color='#10b981')
    axes[1].set_title('Sparsity per Layer (%)')
    axes[1].set_xlabel('Sparsity %')
    axes[1].set_xlim(0, 100)

    plt.suptitle('NeuroEdge — Layer-wise Stats', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/layer_stats.png', dpi=150)
    print("\nPlot saved: results/layer_stats.png")


if __name__ == '__main__':
    np.random.seed(42)
    layers = {
        'conv1':   np.random.randn(32, 1, 3, 3).astype(np.float32) * 0.1,
        'conv2':   np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.1,
        'fc1':     np.random.randn(128, 1600).astype(np.float32) * 0.05,
        'fc2':     np.random.randn(64, 128).astype(np.float32) * 0.05,
        'output':  np.random.randn(10, 64).astype(np.float32) * 0.1,
    }
    report = compute_layer_stats(layers)
    plot_layer_stats(report)
