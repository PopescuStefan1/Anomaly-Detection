import numpy as np
import matplotlib.pyplot as plt
import random

a=2
b=1

x = [[],[],[],[]]
y =  [[],[],[],[]]

for i in range(10000):
    point_type = random.choice([0, 1, 2, 3])
    if point_type == 0:
        var = np.random.normal(i, 10)
        x[0].append(var)
        y[0].append(a * var + b + np.random.normal(i, 10))
    elif point_type == 1:
        var = np.random.normal(i, 1000)
        x[1].append(var)
        y[1].append(a * var + b + np.random.normal(i, 10))
    elif point_type == 2:
        var = np.random.normal(i, 10)
        x[2].append(var)
        y[2].append(a * var + b + np.random.normal(i, 1000))
    else:
        var = np.random.normal(i, 1000)
        x[3].append(var)
        y[3].append(a * var + b + np.random.normal(i, 1000))    


plt.subplot(2, 2, 1)
plt.scatter(x[0], y[0], c='g')
plt.subplot(2, 2, 2)
plt.scatter(x[1], y[1], c='b')
plt.subplot(2, 2, 3)
plt.scatter(x[2], y[2], c='c')
plt.subplot(2, 2, 4)
plt.scatter(x[3], y[3], c='r')
plt.show()

for i in range(4):
    X = np.vstack([np.ones(len(x[i])), x[i]]).T

    # Calculate the hat matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T

    # Leverage scores are the diagonal elements of H
    leverage_scores = np.diag(H)

    # Plotting leverage scores
    plt.subplot(2, 2, i + 1)
    plt.scatter(x[i], y[i], c=leverage_scores, cmap='viridis', s=10)
    plt.colorbar(label="Leverage Score")
    plt.title("Leverage Scores for All Points")

plt.show()
