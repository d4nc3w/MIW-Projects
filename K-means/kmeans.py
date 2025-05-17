import numpy as np
import matplotlib.pyplot as plt
import os

def load_points_from_csv(filepath):
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    return data[:, :2]

def generate_initial_centroids(centroid_count, points):
    centroid_min = np.min(points, axis=0)
    centroid_max = np.max(points, axis=0)
    return np.random.rand(centroid_count, 2) * (centroid_max - centroid_min) + centroid_min

def assign_points_to_centroids(centroids, points):
    distances = []
    for centroid in centroids:
        dist = np.sqrt((points[:, 0] - centroid[0]) ** 2 + (points[:, 1] - centroid[1]) ** 2)
        distances.append(dist)
    distances = np.asarray(distances)
    assignments = np.argmin(distances, axis=0)
    return assignments.reshape(assignments.shape[0], 1)

def assign_colors_to_points(assignments, colormap):
    colors = colormap[assignments.flatten()]
    return colors

def plot_clusters(points, centroids, colors, colormap):
    plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=colormap)
    plt.show()

def recalculate_centroids(points, assignments, centroids):
    new_centroids = []
    for idx, old_centroid in enumerate(centroids):
        assignment = assignments == idx
        assignment = assignment.flatten()
        cluster = points[assignment, :]
        if len(cluster) == 0:
            new_centroids.append(old_centroid)
        else:
            new_centroids.append(np.mean(cluster, axis=0))
    return np.asarray(new_centroids)

def k_means(points, cluster_count, iterations, colormap):
    centroids = generate_initial_centroids(cluster_count, points)
    for i in range(iterations):
        assignments = assign_points_to_centroids(centroids, points)
        colors = assign_colors_to_points(assignments, colormap)
        plot_clusters(points, centroids, colors, colormap)
        centroids = recalculate_centroids(points, assignments, centroids)
    return centroids

def read_datasets():
    for file in os.listdir("clusters"):
        filepath = "clusters/" + file
        print(file)

        with open(filepath, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines[1:]:
            values = line.strip().split(',')
            data.append([float(values[0]), float(values[1]), int(values[2])])

        data = np.array(data)
        points = data[:, :2]
        true_labels = data[:, 2].astype(int)

        cluster_count = len(np.unique(true_labels))
        colormap = np.random.rand(cluster_count, 3)
        k_means(points, cluster_count, 10, colormap)

read_datasets()
