import numpy as np
import matplotlib.pyplot as plt


def quantize_int8(weights):
    scale = np.max(np.abs(weights)) / 127.0
    q = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    dq = q * scale
    error = np.mean(np.abs(weights - dq))
    return q, scale, error


def quantize_int4(weights):
    scale = np.max(np.abs(weights)) / 7.0
    q = np.clip(np.round(weights / scale), -8, 7).astype(np.int8)
    dq = q * scale
    error = np.mean(np.abs(weights - dq))
    return q, scale, error


def mixed_precision_assign(layers: dict, sensitive_layers: list):
    """
    Assign INT8 to sensitive layers, INT4 to rest.
    sensitive_layers: list of layer names that need higher precision.
    """
    print("=" * 58)
    print(" NeuroEdge — Mixed Precision Assignment")
    print("=" * 58)
    print(f"{'Layer':<12} {'Precision':<10} {'Params':>8} {'Error':>10} {'Savings':>10}")
    print("-" * 58)

    report = []
    fp32_total = 0
    quant_total = 0

    for name, w in layers.items():
        params = w.size
        fp32_bytes = params * 4

        if name in sensitive_layers:
            q, scale, error = quantize_int8(w)
            precision = 'INT8'
            quant_bytes = params * 1
        else:
            q, scale, error = quantize_int4(w)
            precision = 'INT4'
            quant_bytes = params // 2

        savings = 100.0 * (1 - quant_bytes / fp32_bytes)
        fp32_total += fp32_bytes
        quant_total += quant_bytes

        print(f"{name:<12} {precision:<10} {params:>8} {error:>10.6f} {savings:>9.1f}%")
        report.append({
            'name': name, 'precision': precision,
            'params': params, 'error': error, 'savings': savings
        })

    overall = 100.0 * (1 - quant_total / fp32_total)
    print("-" * 58)
    print(f"{'TOTAL':<12} {'MIXED':<10} {'':>8} {'':>10} {overall:>9.1f}%")
    print(f"\nFP32: {fp32_total} bytes  →  Mixed: {quant_total} bytes")
    print(f"Overall compression: {fp32_total/quant_total:.2f}x")
    return report


def plot_mixed_precision(report):
    names = [r['name'] for r in report]
    errors = [r['error'] for r in report]
    savings = [r['savings'] for r in report]
    colors = ['#6366f1' if r['precision'] == 'INT8' else '#10b981' for r in report]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(names, errors, color=colors)
    axes[0].set_title('Quantization Error per Layer')
    axes[0].set_ylabel('Mean Abs Error')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(color='#6366f1', label='INT8'),
        Patch(color='#10b981', label='INT4')
    ])

    axes[1].bar(names, savings, color=colors)
    axes[1].set_title('Memory Savings per Layer (%)')
    axes[1].set_ylabel('Savings %')
    axes[1].set_ylim(0, 100)

    plt.suptitle('NeuroEdge — Mixed Precision Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/mixed_precision.png', dpi=150)
    print("\nPlot saved: results/mixed_precision.png")


if __name__ == '__main__':
    np.random.seed(42)
    layers = {
        'conv1':  np.random.randn(32, 1, 3, 3).astype(np.float32) * 0.1,
        'conv2':  np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.1,
        'fc1':    np.random.randn(128, 256).astype(np.float32) * 0.05,
        'fc2':    np.random.randn(10, 128).astype(np.float32) * 0.1,
    }
    # conv1 and fc2 are sensitive (first + last layer)
    sensitive = ['conv1', 'fc2']
    report = mixed_precision_assign(layers, sensitive)
    plot_mixed_precision(report)
