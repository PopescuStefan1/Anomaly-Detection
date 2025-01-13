# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import random
# from sklearn.linear_model import LinearRegression

# regular_graph = nx.random_regular_graph(d=3, n=100)

# plt.figure()
# nx.draw(regular_graph, node_size=50)
# plt.show()

# caveman_graph = nx.connected_caveman_graph(10, 20)

# plt.figure()
# nx.draw(caveman_graph, node_size=50)
# plt.show()

# merged_graph = nx.union(regular_graph, caveman_graph, rename=('R-', 'C-'))
# for _ in range(50):
#     node1 = random.choice(list(regular_graph.nodes()))
#     node2 = random.choice(list(caveman_graph.nodes()))
#     merged_graph.add_edge(f'R-{node1}', f'C-{node2}')

# plt.figure()
# nx.draw(merged_graph, node_size=50)
# plt.show()

# features = []
# nodes = list(merged_graph.nodes)

# for node in nodes:
#     ego_graph = nx.ego_graph(merged_graph, node)
#     Ni = ego_graph.number_of_nodes() - 1
#     Ei = ego_graph.number_of_edges()
#     features.append([Ni, Ei])

# features = np.array(features)
# Ni = features[:, 0].reshape(-1, 1)
# Ei = features[:, 1]

# log_Ni = np.log(Ni)
# log_Ei = np.log(Ei)

# reg = LinearRegression().fit(log_Ni, log_Ei)
# theta = reg.coef_[0]
# C = np.exp(reg.intercept_)

# anomaly_scores = []
# for i in range(len(Ei)):
#     yi = Ei[i]
#     xi = Ni[i][0]
#     predicted_y = C * (xi ** theta)
#     score = (max(yi, predicted_y) / min(yi, predicted_y)) * np.log(abs(yi - predicted_y) + 1)
#     anomaly_scores.append(score)

# anomaly_scores = np.array(anomaly_scores)
# top_10_indices = anomaly_scores.argsort()[-10:][::-1]
# top_10_nodes = [nodes[i] for i in top_10_indices]

# color_map = []
# for node in merged_graph.nodes:
#     if node in top_10_nodes:
#         color_map.append('red')  
#     else:
#         color_map.append('blue') 

# plt.figure()
# nx.draw(
#     merged_graph,
#     node_color=color_map,
#     node_size=50,
# )
# plt.show()

import networkx as nx
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

G1 = nx.random_regular_graph(d=3, n=100)  
G2 = nx.random_regular_graph(d=5, n=100)  

G = nx.union(G1, G2, rename=('G1-', 'G2-'))

# for _ in range(10):  
#     node_from_G1 = random.choice(list(G1.nodes()))
#     node_from_G2 = random.choice(list(G2.nodes()))
#     G.add_edge(f'G1-{node_from_G1}', f'G2-{node_from_G2}')

for edge in G.edges:
    G.add_edge(edge[0], edge[1], weight=1)

node1, node2 = random.sample(list(G.nodes()), 2)

for neighbor in G.neighbors(node1):
    G[node1][neighbor]['weight'] += 10
for neighbor in G.neighbors(node2):
    G[node2][neighbor]['weight'] += 10

features = []
for node in G.nodes:
    EG = nx.ego_graph(G, node)
    Ni = EG.number_of_nodes() - 1  
    Ei = EG.number_of_edges()
    Wi = EG.size()
    features.append([Ni, Ei, Wi])

features = np.array(features)
Ni = features[:, 0].reshape(-1, 1)
Ei = features[:, 1]
Wi = features[:, 2]

log_Ni = np.log(Ni)
log_Ei = np.log(Ei)

reg = LinearRegression().fit(log_Ni, log_Ei)
theta = reg.coef_[0]
C = np.exp(reg.intercept_)

anomaly_scores = []
for i in range(len(Ei)):
    yi = Ei[i]
    xi = Ni[i][0]
    wi = Wi[i]
    predicted_y = C * (xi ** theta)
    score = (max(yi, predicted_y) / min(yi, predicted_y)) * np.log(abs(yi - predicted_y) + 1) + wi / 100  
    anomaly_scores.append(score)

anomaly_scores = np.array(anomaly_scores)
top_4_indices = anomaly_scores.argsort()[-4:][::-1]
top_4_nodes = [list(G.nodes)[i] for i in top_4_indices]

color_map = []
for node in G.nodes:
    if node in top_4_nodes:
        color_map.append('red')  
    else:
        color_map.append('blue') 

plt.figure()
nx.draw(
    G,
    node_color=color_map,
    node_size=50
)
plt.show()
