from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(n_samples=500, centers=[[0, 0]])

plt.scatter(X[:, 0], X[:, 1])
plt.show()

random_vectors = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=5)

print(random_vectors)

hist_data = []
bin_count = 10

for i, unit_vector in enumerate(random_vectors):
    projections = X @ unit_vector
    
    range = (projections.min() - 1, projections.max() + 1)
    hist_values = np.histogram(projections, bins=bin_count, range=range)
    hist_probs, bin_edges = np.histogram(projections, range=range, density=True)
    
    hist_data.append((hist_probs, bin_edges))
    
    plt.figure()
    plt.hist(projections, bins=bin_count, range=range, edgecolor='black')
    plt.title(f"Histogram of Projections on Unit Vector {i+1}")
    plt.show()
    
def calculate_anomaly_score(sample, hist_data, random_vectors):
    scores = []
    
    for (bin_probs, bin_edges), unit_vector in zip(hist_data, random_vectors):
        sample_projection = sample @ unit_vector
        
        bin_index = np.digitize(sample_projection, bin_edges) - 1
        
        bin_index = np.clip(bin_index, 0, len(bin_probs) - 1)
        
        probability = bin_probs[bin_index]
        scores.append(probability)
    
    anomaly_score = np.mean(scores)
    return anomaly_score

### TRAIN DATA

anomaly_scores = np.array([calculate_anomaly_score(sample, hist_data, random_vectors) for sample in X])

threshold = np.percentile(anomaly_scores, 10)
anomalies = X[anomaly_scores < threshold]

print("Anomaly scores:\n", anomaly_scores)
print("Anomalies:\n", anomalies)

plt.scatter(X[:, 0], X[:, 1], color='blue')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red')
plt.title("Train data:")
plt.show()

### TEST DATA

X = np.random.uniform(low=[-3, -3], high=[3, 3], size=(500, 2))
anomaly_scores = np.array([calculate_anomaly_score(sample, hist_data, random_vectors) for sample in X])

threshold = np.percentile(anomaly_scores, 10)
anomalies = X[anomaly_scores < threshold]

print("Anomaly scores:\n", anomaly_scores)
print("Anomalies:\n", anomalies)

plt.scatter(X[:, 0], X[:, 1], color='blue')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red')
plt.title("Test data")
plt.show()