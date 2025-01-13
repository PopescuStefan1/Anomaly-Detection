import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt

dataset = np.loadtxt('datasets/ca-AstroPh.txt')[:1500]

G = nx.Graph()
for line in dataset:
    u, v = line
    if G.has_edge(u, v):
        G[u][v]['weight'] += 1
    else:
        G.add_edge(u, v, weight=1)

features = []
for node in G.nodes:
    EG = nx.generators.ego_graph(G, node)
    Ni = EG.number_of_nodes() - 1
    Ei = EG.number_of_edges()
    adj_matrix = nx.to_numpy_array(EG, weight='weight')
    eigenvalues = np.linalg.eigvals(adj_matrix)
    principal_eigenvalue = max(eigenvalues)

    G.nodes[node]['no_of_neighbors'] = Ni
    G.nodes[node]['no_of_edges'] = Ei
    G.nodes[node]['size'] = EG.size()
    G.nodes[node]['principal_eigenvalue'] = principal_eigenvalue

    features.append([Ni, Ei])

features = np.array(features)
Ni = features[:, 0].reshape(-1, 1)
Ei = features[:, 1]

log_Ni = np.log(Ni)
log_Ei = np.log(Ei)
reg = LinearRegression().fit(log_Ni, log_Ei)
theta = reg.coef_[0]
C = np.exp(reg.intercept_)

normalized_scores = []
for i in range(len(Ei)):
    yi = Ei[i]
    xi = Ni[i][0]
    predicted_y = C * (xi ** theta)
    score = (max(yi, predicted_y) / min(yi, predicted_y)) * np.log(abs(yi - predicted_y) + 1)
    normalized_scores.append(score)

normalized_scores = np.array(normalized_scores)
normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())

lof = LocalOutlierFactor(n_neighbors=20)
lof.fit_predict(features)
lof_scores = lof.negative_outlier_factor_
lof_scores = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

combined_scores = normalized_scores + lof_scores

combined_anomaly_scores = {node: score for node, score in zip(G.nodes, combined_scores)}
sorted_scores = sorted(combined_anomaly_scores.items(), key=lambda x: x[1], reverse=True)

top_10_nodes = [node for node, score in sorted_scores[:10]]
color_map = []
for node in G.nodes:
    if node in top_10_nodes:
        color_map.append('red')  
    else:
        color_map.append('blue')  

plt.figure()
nx.draw(
    G,
    node_color=color_map,
    node_size=50,
)
plt.show()
