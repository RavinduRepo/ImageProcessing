# Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv('fashion_mnist/fashion_mnist_train.csv')
test_data = pd.read_csv('fashion_mnist/fashion_mnist_test.csv')

# Preprocess data
X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test = test_data.iloc[:, 0].values

# Visualize data
def visualize_data(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.show()

visualize_data(X_train, y_train)

# ANN Model
ann_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=10)
ann_model.save('ANN_model.h5')

# Visualize classified images
def visualize_classified_images(model, X, y):
    predictions = model.predict(X)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(f"True: {y[i]}, Pred: {np.argmax(predictions[i])}")
    plt.show()

# Load the saved ANN model
loaded_ann_model = tf.keras.models.load_model('ANN_model.h5')

# Visualize classified images using the loaded model
visualize_classified_images(loaded_ann_model, X_test, y_test)

# CNN Model with even larger kernel size
cnn_model = Sequential([
    Conv2D(64, (13, 13), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (13, 13), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=10)
cnn_model.save('CNN_model.h5')

# Load the saved CNN model
loaded_cnn_model = tf.keras.models.load_model('CNN_model.h5')

# Visualize classified images using the loaded CNN model
visualize_classified_images(loaded_cnn_model, X_test, y_test)

# Visualize CNN layer kernels
def visualize_cnn_kernels(model, layer_index):
    layer = model.layers[layer_index]
    if 'conv' not in layer.name:
        print(f"Layer {layer_index} is not a convolutional layer.")
        return
    
    kernels, biases = layer.get_weights()
    n_kernels = kernels.shape[-1]
    
    plt.figure(figsize=(10, 10))
    for i in range(min(n_kernels, 64)):  # Visualize up to 64 kernels
        plt.subplot(8, 8, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(kernels[:, :, 0, i], cmap='viridis')
        plt.xlabel(f'Kernel {i}')
    plt.show()

# Visualize kernels of intermediate layers
def visualize_intermediate_kernels(model):
    for i, layer in enumerate(model.layers):
        if 'conv' in layer.name:
            print(f"Visualizing kernels of layer {i} ({layer.name})")
            visualize_cnn_kernels(model, i)

# Example usage: Visualize kernels of all convolutional layers
visualize_intermediate_kernels(loaded_cnn_model)