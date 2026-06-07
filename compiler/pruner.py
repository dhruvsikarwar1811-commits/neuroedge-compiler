# NeuroEdge — Pruner Module
# Weight pruning — near-zero weights remove karna
import numpy as np
import tensorflow as tf

def magnitude_pruning(model, sparsity=0.3):
    print(f"\n[NeuroEdge] Starting Pruning (sparsity={sparsity*100:.0f}%)...")
    total_weights = 0
    total_pruned = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        new_weights = []
        for w in weights:
            threshold = np.percentile(np.abs(w), sparsity * 100)
            mask = np.abs(w) > threshold
            pruned_w = w * mask
            new_weights.append(pruned_w)
            total_pruned += np.sum(~mask)
            total_weights += w.size
        layer.set_weights(new_weights)
    actual_sparsity = total_pruned / total_weights
    print(f"[NeuroEdge] Weights pruned  : {total_pruned:,}")
    print(f"[NeuroEdge] Total weights   : {total_weights:,}")
    print(f"[NeuroEdge] Actual sparsity : {actual_sparsity:.1%}")
    return model, actual_sparsity

if __name__ == "__main__":
    model = tf.keras.models.load_model('models/mnist_cnn.h5')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0
    loss, acc_before = model.evaluate(x_test, y_test, verbose=0)
    print(f"[NeuroEdge] Accuracy BEFORE pruning: {acc_before*100:.2f}%")
    pruned_model, sparsity = magnitude_pruning(model, sparsity=0.3)
    loss, acc_after = pruned_model.evaluate(x_test, y_test, verbose=0)
    print(f"[NeuroEdge] Accuracy AFTER  pruning: {acc_after*100:.2f}%")
    print(f"[NeuroEdge] Accuracy drop: {(acc_before-acc_after)*100:.2f}%")
