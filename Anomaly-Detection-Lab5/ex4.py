import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

noise_factor = 0.35
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=tf.shape(x_train), dtype=tf.float32)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=tf.shape(x_test), dtype=tf.float32)

x_train_noisy = tf.clip_by_value(x_train_noisy, 0.0, 1.0)
x_test_noisy = tf.clip_by_value(x_test_noisy, 0.0, 1.0)

class ConvolutionalAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(4, (3, 3), activation='relu', strides=2, padding='same')
        ])
        self.decoder = models.Sequential([
            layers.Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = ConvolutionalAutoencoder()

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test_noisy, x_test)
)

reconstructed_from_clean = autoencoder.predict(x_test[:5])
reconstructed_from_noisy = autoencoder.predict(x_test_noisy[:5])

plt.figure(figsize=(15, 8))

for i in range(5):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Original", fontsize=12)
    
    plt.subplot(4, 5, i + 6)
    plt.imshow(x_test_noisy[i].numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Noisy", fontsize=12)
    
    plt.subplot(4, 5, i + 11)
    plt.imshow(reconstructed_from_clean[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Reconstructed\nfrom Original", fontsize=12)
    
    plt.subplot(4, 5, i + 16)
    plt.imshow(reconstructed_from_noisy[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel("Reconstructed\nfrom Noisy", fontsize=12)

plt.tight_layout()
plt.show()
