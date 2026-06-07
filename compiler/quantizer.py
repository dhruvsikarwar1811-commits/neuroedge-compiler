# NeuroEdge — Quantizer Module
# Float32 to INT8 conversion
import tensorflow as tf
import numpy as np
import os

def quantize_model(model_path, x_train):
    print("\n[NeuroEdge] Starting Quantization...")
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data_gen():
        for i in range(100):
            yield [x_train[i:i+1].astype(np.float32)]
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()
    os.makedirs('models', exist_ok=True)
    with open('models/mnist_cnn_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    original_size = os.path.getsize(model_path) / 1024
    quantized_size = len(tflite_model) / 1024
    reduction = original_size / quantized_size
    print(f"[NeuroEdge] Original size  : {original_size:.1f} KB")
    print(f"[NeuroEdge] Quantized size : {quantized_size:.1f} KB")
    print(f"[NeuroEdge] Size reduction : {reduction:.1f}x smaller")
    print("[NeuroEdge] Saved: models/mnist_cnn_quantized.tflite")
    return tflite_model

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from models.train_model import train_and_save
    model, x_train, x_test = train_and_save()
    quantize_model('models/mnist_cnn.h5', x_train)
