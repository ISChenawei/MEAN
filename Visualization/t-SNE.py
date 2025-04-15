import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Custom color lists
color_list = [
    (0.1, 0.1, 0.1, 1.0),  # r, g, b
    (0.5, 0.5, 0.5, 1.0),
    (1.0, 0.6, 0.1, 1.0),
    (1.0, 0.0, 0.0, 1.0),
    (1.0, 0.1, 0.7, 1.0),
    (0.9, 0.2, 0.4, 1.0),
    (0.8, 0.2, 1.0, 1.0),
    (0.8, 0.3, 0.2, 1.0),
    (0.7, 0.5, 0.3, 1.0),
    (0.7, 0.9, 0.4, 1.0),
    (0.7, 0.3, 0.8, 1.0),
    (0.0, 1.0, 0.0, 1.0),
    (0.4, 1.0, 0.6, 1.0),
    (0.3, 0.8, 0.5, 1.0),
    (0.1, 0.8, 1.0, 1.0),
    (0.5, 0.7, 0.9, 1.0),
    (0.4, 0.8, 0.3, 1.0),
    (0.5, 0.7, 0.4, 1.0),
    (0.2, 0.6, 0.8, 1.0),
    (0.1, 0.1, 1.0, 1.0),
    (0.3, 0.3, 0.9, 1.0),
    (0.6, 0.1, 0.4, 1.0),
]

# Load saved feature data
def load_features_from_mat(file_path):
    data = scipy.io.loadmat(file_path)
    query_features = data['query_features']
    gallery_features = data['gallery_features']
    query_labels = data['query_labels'].flatten()  # 1D array
    gallery_labels = data['gallery_labels'].flatten()  # 1D array
    return query_features, gallery_features, query_labels, gallery_labels

# Select up to 20 categories for visualization
def filter_classes(query_features, gallery_features, query_labels, gallery_labels, num_classes=10):
    unique_labels = np.unique(np.concatenate([query_labels, gallery_labels]))
    selected_labels = np.random.choice(unique_labels, num_classes, replace=False)  # Randomly select num_classes categories

    query_mask = np.isin(query_labels, selected_labels)
    gallery_mask = np.isin(gallery_labels, selected_labels)

    filtered_query_features = query_features[query_mask]
    filtered_query_labels = query_labels[query_mask]
    filtered_gallery_features = gallery_features[gallery_mask]
    filtered_gallery_labels = gallery_labels[gallery_mask]

    return filtered_query_features, filtered_gallery_features, filtered_query_labels, filtered_gallery_labels

# Sample each category of gallery, limiting the number of points per category
def sample_gallery_per_class(gallery_features, gallery_labels, max_gallery_points_per_class=10):
    unique_labels = np.unique(gallery_labels)
    sampled_features = []
    sampled_labels = []

    for label in unique_labels:
        indices = np.where(gallery_labels == label)[0]
        if len(indices) > max_gallery_points_per_class:
            sampled_indices = np.random.choice(indices, max_gallery_points_per_class, replace=False)
        else:
            sampled_indices = indices
        sampled_features.append(gallery_features[sampled_indices])
        sampled_labels.append(gallery_labels[sampled_indices])

    # Merge sampled gallery features and tags
    sampled_features = np.vstack(sampled_features)
    sampled_labels = np.hstack(sampled_labels)

    return sampled_features, sampled_labels

# Filter out data in the -1 category
def filter_invalid_classes(features, labels):
    valid_mask = labels != -1
    return features[valid_mask], labels[valid_mask]

# Use t-SNE for dimensionality reduction and visualization
def visualize_tsne(query_features, gallery_features, query_labels, gallery_labels):
    # Splicing query and gallery features
    features = np.vstack((query_features, gallery_features))
    labels = np.hstack((query_labels, gallery_labels))

    # Use t-SNE to reduce features to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # Assign colors
    unique_labels = np.unique(labels)
    if len(unique_labels) > len(color_list):
        raise ValueError("The number of unique classes exceeds the provided color list length.")

    # Plotting t-SNE results
    plt.figure(figsize=(10, 10))

    num_query = len(query_features)
    for i, label in enumerate(unique_labels):
        color = color_list[i % len(color_list)]  # 使用自定义颜色
        # query is marked with a pentagram
        query_indices = np.where((labels == label) & (np.arange(len(labels)) < num_query))[0]
        plt.scatter(features_tsne[query_indices, 0], features_tsne[query_indices, 1],
                    color=color, marker='*', s=300, label=f'Query Class {int(label)}')

        # gallery's markers are dots
        gallery_indices = np.where((labels == label) & (np.arange(len(labels)) >= num_query))[0]
        plt.scatter(features_tsne[gallery_indices, 0], features_tsne[gallery_indices, 1],
                    color=color, marker='x', s=20, label=f'Gallery Class {int(label)}')

    # plt.title('t-SNE visualization of query and gallery features')
    # plt.legend()
    plt.savefig("OUTPUT1",dpi = 1200)
    plt.show()


file_path = 'features.mat'  # Replace with the actual .mat file path
query_features, gallery_features, query_labels, gallery_labels = load_features_from_mat(file_path)


filtered_query_features, filtered_gallery_features, filtered_query_labels, filtered_gallery_labels = filter_classes(
    query_features, gallery_features, query_labels, gallery_labels, num_classes=20)

sampled_gallery_features, sampled_gallery_labels = sample_gallery_per_class(filtered_gallery_features, filtered_gallery_labels, max_gallery_points_per_class=20)

filtered_query_features, filtered_query_labels = filter_invalid_classes(filtered_query_features, filtered_query_labels)
sampled_gallery_features, sampled_gallery_labels = filter_invalid_classes(sampled_gallery_features, sampled_gallery_labels)


visualize_tsne(filtered_query_features, sampled_gallery_features, filtered_query_labels, sampled_gallery_labels)

