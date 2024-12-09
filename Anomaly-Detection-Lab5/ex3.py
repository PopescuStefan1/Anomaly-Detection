import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
from keras import layers, Model, Sequential
import matplotlib.pyplot as plt

dataset = scipy.io.loadmat('datasets/shuttle.mat')
X = dataset['X']
y = dataset['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)  

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(8, activation='relu'),
            layers.Dense(5, activation='relu'),
            layers.Dense(3, activation='relu')  
        ])
        self.decoder = Sequential([
            layers.Dense(5, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(X_train.shape[1], activation='sigmoid')  
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_train_standardized,
    X_train_standardized,  
    epochs=100,
    batch_size=1024,
    validation_data=(X_test_standardized, X_test_standardized),
    verbose=1
)

train_reconstructed = autoencoder.predict(X_train_standardized)
test_reconstructed = autoencoder.predict(X_test_standardized)

train_reconstruction_error = np.mean(np.square(X_train_standardized - train_reconstructed), axis=1)
test_reconstruction_error = np.mean(np.square(X_test_standardized - test_reconstructed), axis=1)

contamination_rate = 0.2
threshold = np.quantile(train_reconstruction_error, 1 - contamination_rate)

train_predictions = (train_reconstruction_error > threshold).astype(int)
test_predictions = (test_reconstruction_error > threshold).astype(int)

y_train_binary = (y_train > 1).astype(int)
y_test_binary = (y_test > 1).astype(int)

train_balanced_accuracy = balanced_accuracy_score(y_train_binary, train_predictions)
test_balanced_accuracy = balanced_accuracy_score(y_test_binary, test_predictions)

print(f"Train Balanced Accuracy: {train_balanced_accuracy}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy}")

plt.figure()
plt.hist(train_reconstruction_error, label='Train Reconstruction Error')
plt.hist(test_reconstruction_error, label='Test Reconstruction Error')
plt.legend()
plt.show()
