import numpy as np
import matplotlib.pyplot as plt

POINT_COUNT = 100
CLUSTER_COUNT = 4
CENTROID_MIN = np.asarray([-20, -30])
CENTROID_MAX = np.asarray([45, 50])
NOISE = 4
ITERATIONS = 10

def generate_random_clusters(centroid_min, centroid_max, noise_level, centroid_count, point_count):
    centroids = np.random.rand(centroid_count, 2) * (centroid_max - centroid_min) + centroid_min
    assignments = np.random.randint(0, centroid_count, point_count)
    assignments = assignments.reshape(assignments.shape[0], 1)
    offsets = (np.random.rand(point_count, 2)*2-1)*noise_level
    points = centroids[assignments]
    points = points.reshape(points.shape[0], points.shape[2])
    points += offsets
    return (points)

points = generate_random_clusters(CENTROID_MIN, CENTROID_MAX, NOISE, CLUSTER_COUNT, POINT_COUNT)
plt.scatter(points[:, 0], points[:, 1])
plt.show()

COLORMAP = np.random.rand(CLUSTER_COUNT, 3)
print(COLORMAP)

# IMPLEMENTING K-MEANS
def generate_initial_centroids(centroid_count, centroid_max, centroid_min):
    return np.random.rand(centroid_count, 2) * (centroid_max - centroid_min) + centroid_min

centroids = generate_initial_centroids(CLUSTER_COUNT, CENTROID_MAX, CENTROID_MIN)
plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c=COLORMAP)
plt.show()

def assign_points_to_centroids(centroids, points):
    distances = []
    for centroid in centroids:
        dist = np.sqrt((points[:, 0] - centroid[0])**2 + (points[:, 1] - centroid[1])**2)
        distances.append(dist)
    distances = np.asarray(distances)
    assignments = np.argmin(distances, axis=0)
    return assignments.reshape(assignments.shape[0], 1)

assignments = assign_points_to_centroids(centroids, points)

def assign_colors_to_points(assignments, colormap):
    colors = COLORMAP[assignments]
    return colors.reshape(colors.shape[0], colors.shape[2])

def plot_clusters(points, centroids, colors, colormap):
    plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=colormap)
    plt.show()

colors = assign_colors_to_points(assignments, COLORMAP)
plot_clusters(points, centroids, colors, COLORMAP)

def recalculate_centroids(points, assignments, old_centroids):
    new_centroids = []
    for idx, old_centroid in enumerate(centroids):
        assignment = assignments==idx
        assignment = assignment.flatten()
        cluster = points[assignment, :]
        if len(cluster) == 0:
            new_centroids.append(old_centroid)
        else:
            new_centroids.append(np.mean(cluster, axis=0))
    return np.asarray(new_centroids)

new_centroids = recalculate_centroids(points, assignments, COLORMAP)
plot_clusters(points, new_centroids, colors, COLORMAP)

def k_means(points, cluster_count, iterations, colormap):
    min_position = np.min(points, axis=0)
    max_position = np.max(points, axis=0)
    centroids = generate_initial_centroids(cluster_count, max_position, min_position)
    assignments = assign_points_to_centroids(centroids, points)
    colors = assign_colors_to_points(assignments, COLORMAP)
    plot_clusters(points, centroids, colors, COLORMAP)
    while iterations > 0:
        iterations -= 1
        recalculate_centroids(points, assignments, COLORMAP)
        assignments = assign_points_to_centroids(centroids, points)
        colors = assign_colors_to_points(assignments, COLORMAP)
        plot_clusters(points, centroids, colors, COLORMAP)
    return centroids

MANY_CLUSTERS = np.random.rand(CLUSTER_COUNT, 3)
final_centroids = k_means(points, CLUSTER_COUNT, ITERATIONS, COLORMAP)
print(final_centroids)