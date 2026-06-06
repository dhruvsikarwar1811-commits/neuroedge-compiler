# NeuroEdge — Model Analyzer
# Phase 1: Neural network ka structure analyze karna
# Author: Yuvraj Sikarwar

import tensorflow as tf
import json
import os

def estimate_hw_operations(layer):
    """Hardware operations estimate karta hai har layer ke liye"""
    if isinstance(layer, tf.keras.layers.Conv2D):
        config = layer.get_config()
        return {
            "type": "conv",
            "filters": config['filters'],
            "kernel": config['kernel_size'][0]
        }
    elif isinstance(layer, tf.keras.layers.Dense):
        config = layer.get_config()
        return {
            "type": "dense",
            "units": config['units']
        }
    elif isinstance(layer, tf.keras.layers.MaxPooling2D):
        return {"type": "maxpool"}
    elif isinstance(layer, tf.keras.layers.Flatten):
        return {"type": "flatten"}
    else:
        return {"type": "other"}

def analyze_model(model):
    """
    Model ka poora analysis karta hai:
    - Total parameters
    - Har layer ki details
    - Hardware cost estimate
    """
    print("\n" + "="*50)
    print("  NeuroEdge — Model Analyzer")
    print("="*50)

    analysis = {
        "total_params": int(model.count_params()),
        "total_layers": len(model.layers),
        "layers": []
    }

    print(f"\n Total Parameters : {analysis['total_params']:,}")
    print(f" Total Layers     : {analysis['total_layers']}")
    print("\n Layer-wise Breakdown:")
    print("-"*50)

    for i, layer in enumerate(model.layers):
        layer_info = {
            "index": i,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(layer.output_shape),
            "params": int(layer.count_params()),
            "hw_ops": estimate_hw_operations(layer)
        }
        analysis["layers"].append(layer_info)

        print(f" [{i}] {layer.__class__.__name__:<20} "
              f"params={layer.count_params():>6,} "
              f"output={layer.output_shape}")

    # JSON file mein save karo
    os.makedirs("compiler", exist_ok=True)
    with open("compiler/model_profile.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\n" + "="*50)
    print(" Profile saved: compiler/model_profile.json")
    print("="*50 + "\n")

    return analysis

if __name__ == "__main__":
    # Test: ek simple model banao aur analyze karo
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation='relu',
                               input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    analyze_model(model)