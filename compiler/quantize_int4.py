import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def quantize_int4(weights):
    """INT4 quantization — 2x more compression than INT8"""
    scale = np.max(np.abs(weights)) / 7.0
    quantized = np.clip(np.round(weights / scale), -8, 7).astype(np.int8)
    dequantized = quantized * scale
    error = np.mean(np.abs(weights - dequantized))
    return quantized, scale, error

def compare_quantization(weights):
    """Compare FP32 vs INT8 vs INT4"""
    # INT8
    scale8 = np.max(np.abs(weights)) / 127.0
    q8 = np.clip(np.round(weights / scale8), -128, 127).astype(np.int8)
    err8 = np.mean(np.abs(weights - q8 * scale8))

    # INT4
    q4, scale4, err4 = quantize_int4(weights)

    print("=" * 45)
    print(" NeuroEdge — Quantization Comparison")
    print("=" * 45)
    print(f"FP32  size: {weights.nbytes} bytes")
    print(f"INT8  size: {q8.nbytes} bytes  | error: {err8:.6f}")
    print(f"INT4* size: {q8.nbytes//2} bytes  | error: {err4:.6f}")
    print(f"INT4 compression: {weights.nbytes // (q8.nbytes//2)}x vs FP32")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(weights.flatten(), bins=50, color='#0ea5e9')
    axes[0].set_title('FP32 Weights')
    axes[1].hist(q8.flatten(), bins=50, color='#7c3aed')
    axes[1].set_title('INT8 Quantized')
    axes[2].hist(q4.flatten(), bins=50, color='#10b981')
    axes[2].set_title('INT4 Quantized')
    plt.suptitle('NeuroEdge — Weight Distribution', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/quantization_comparison.png', dpi=150)
    print("Plot saved: results/quantization_comparison.png")

if __name__ == '__main__':
    np.random.seed(42)
    weights = np.random.randn(128, 128).astype(np.float32)
    compare_quantization(weights)
