# NeuroEdge — CNN Model Training
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_and_save():
    print("[NeuroEdge] Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
    x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[NeuroEdge] Test Accuracy: {test_acc*100:.2f}%")
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    model.save('models/mnist_cnn.h5')
    print("[NeuroEdge] Model saved!")
    return model, x_train, x_test

if __name__ == "__main__":
    train_and_save()
