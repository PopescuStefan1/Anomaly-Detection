import numpy as np
import matplotlib.pyplot as plt

random_vectors = np.random.multivariate_normal(mean=[5, 10, 2], cov=[[3, 2, 2], [2, 10, 1], [2, 1, 2]], size=500)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(random_vectors[:, 0], random_vectors[:, 1], random_vectors[:, 2])

plt.show()

mean_vector = np.mean(random_vectors, axis=0)
centered_data = random_vectors - mean_vector

cov_matrix = np.cov(centered_data, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]

cumulative_explained_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

plt.figure()
plt.bar(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues / np.sum(sorted_eigenvalues), color='blue')
plt.step(range(1, len(sorted_eigenvalues) + 1), cumulative_explained_variance, where='mid', color='orange')
plt.show()

sorted_eigenvectors = eigenvectors[:, sorted_indices]

projected_data = centered_data @ sorted_eigenvectors

third_pc_values = projected_data[:, 2] 
mean_third_pc = np.mean(third_pc_values)
threshold_third_pc = np.quantile(third_pc_values, 0.9)  
labels_third_pc = np.abs(third_pc_values - mean_third_pc) > threshold_third_pc

second_pc_values = projected_data[:, 1]
mean_second_pc = np.mean(second_pc_values)
threshold_second_pc = np.quantile(second_pc_values, 0.9)
labels_second_pc = np.abs(second_pc_values - mean_second_pc) > threshold_second_pc

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(random_vectors[:, 0], random_vectors[:, 1], random_vectors[:, 2], 
            c=['red' if label else 'blue' for label in labels_third_pc], alpha=0.6)

ax2 = fig.add_subplot(122, projection='3d') 
ax2.scatter(random_vectors[:, 0], random_vectors[:, 1], random_vectors[:, 2], 
            c=['green' if label else 'blue' for label in labels_second_pc], alpha=0.6)

ax1.title.set_text('3rd principal')
ax2.title.set_text('2nd principal')
plt.show()

std_devs = np.std(projected_data, axis=0)  
normalized_distances = np.abs(projected_data / std_devs)

contamination_rate = 0.1
threshold = np.quantile(normalized_distances, 1 - contamination_rate)

outlier_labels = np.any(normalized_distances > threshold, axis=1)  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(random_vectors[:, 0], random_vectors[:, 1], random_vectors[:, 2], 
           c=['red' if label else 'blue' for label in outlier_labels])

plt.title('Centroid')
plt.show()
