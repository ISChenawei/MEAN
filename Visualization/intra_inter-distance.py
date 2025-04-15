import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load saved feature data
def load_features_from_mat(file_path):
    data = scipy.io.loadmat(file_path)
    query_features = data['query_features']
    gallery_features = data['gallery_features']
    query_labels = data['query_labels'].flatten()  # 1D array
    gallery_labels = data['gallery_labels'].flatten()  # 1D array
    return query_features, gallery_features, query_labels, gallery_labels

# Calculate the Cosine distance
def cosine_distance(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1 - np.dot(a_norm, b_norm.T)

file_path = 'features.mat'  # Replace with the path to your .mat file
query_features, gallery_features, query_labels, gallery_labels = load_features_from_mat(file_path)

query_features = torch.FloatTensor(query_features)
gallery_features = torch.FloatTensor(gallery_features)
query_labels = torch.FloatTensor(query_labels)
gallery_labels = torch.FloatTensor(gallery_labels)

# Create a mask for compartmentalizing intra-class and inter-class distances
mask = query_labels.expand(len(gallery_labels), len(query_labels)).eq(gallery_labels.expand(len(query_labels), len(gallery_labels)).t())

# Calculate the Cosine distance matrix
distmat = torch.FloatTensor(cosine_distance(gallery_features.numpy(), query_features.numpy()))  # Cosine distance

# Intra-class and inter-class distances
intra = distmat[mask]
inter = distmat[~mask]

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots()
b = np.linspace(0.3, 0.7, num=200)  # 距离范围

ax.hist(intra.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color='blue', density=True, label='Intra-class')
ax.hist(inter.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color='green', density=True, label='Inter-class')

ax.set_xlabel('Feature Distance')
ax.set_ylabel('Frequency')
ax.legend()

# save figure
# fig.savefig('feature_distance_histogram.svg', dpi=1000, format='svg')
plt.show()
