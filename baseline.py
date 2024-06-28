import pandas as pd
import numpy as np
from multiprocessing import Pool
from fastdtw import fastdtw
import os
from scipy.spatial.distance import euclidean
from simplification.cutil import simplify_coords
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from multiprocessing import Pool
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from scipy.spatial.distance import directed_hausdorff
from fastdtw import fastdtw
from simplification.cutil import simplify_coords
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor





def simplify_trajectory(trajectory, epsilon=0.0005):
    # Simplify the trajectory coordinates.
    coords = np.array(list(zip(trajectory['lon'], trajectory['lat'])))
    simplified_coords = simplify_coords(coords, epsilon)

    # Function to find the closest index in the original coordinates for each simplified coordinate
    def find_closest(original, simplified_point):
        distances = np.sqrt((original[:, 0] - simplified_point[0])**2 + (original[:, 1] - simplified_point[1])**2)
        return np.argmin(distances)

    # Find the indices of the closest original points to the simplified points
    simplified_indices = [find_closest(coords, point) for point in simplified_coords]

    # Create the simplified trajectory using the closest indices to keep speed and course consistent
    simplified_trajectory = {
        'lon': [trajectory['lon'][i] for i in simplified_indices],
        'lat': [trajectory['lat'][i] for i in simplified_indices],
        'speed': [trajectory['speed'][i] for i in simplified_indices],
        'course': [trajectory['course'][i] for i in simplified_indices]
    }

    return simplified_trajectory


def convert_to_dataframe(data):
    if isinstance(data, dict):
        return pd.DataFrame(data)
    return data

def compute_dtw_distance(index_pair, all_trajectories):
    i, j = index_pair
    traj1 = convert_to_dataframe(all_trajectories[i])
    traj2 = convert_to_dataframe(all_trajectories[j])

    # Assuming traj1 and traj2 are DataFrames with columns ['lat', 'lon', 'speed', 'course']
    distance, path = fastdtw(traj1[['lat', 'lon', 'speed', 'course']].to_numpy(),
                             traj2[['lat', 'lon', 'speed', 'course']].to_numpy(),
                             dist=euclidean)
    return (i, j, distance)


def parallel_distance_matrix(all_trajectories, num_processes):
    # Generate all unique pairs of indices for the trajectories
    from itertools import combinations
    index_pairs = list(combinations(range(len(all_trajectories)), 2))

    # Use a multiprocessing pool to compute distances in parallel
    with Pool(processes=num_processes) as pool:
        result = pool.starmap(compute_dtw_distance, [(pair, all_trajectories) for pair in index_pairs])

    # Create an empty distance matrix
    n = len(all_trajectories)
    distance_matrix = np.zeros((n, n))

    # Fill in the computed distances for both upper and lower triangular parts
    for (i, j, dist) in result:
        distance_matrix[i][j] = dist
        distance_matrix[j][i] = dist  # Ensure the matrix is symmetric

    return distance_matrix


def save_distance_matrix(matrix, filename):
    np.save(filename, matrix)

# Load the distance matrix from a .npy file
def load_distance_matrix(filename):
    return np.load(filename, allow_pickle=False)

# Main function to compute or load the distance matrix
def compute_or_load_distances(all_trajectories, filename='all_dist_matrix.npy', num_processes=16):
    if (os.path.exists(filename)):
        print("Loading existing distance matrix.")
        return load_distance_matrix(filename)
    else:
        print("Computing distance matrix.")
        matrix = parallel_distance_matrix(all_trajectories, num_processes)
        save_distance_matrix(matrix, filename)
        return matrix
    
def visualize_outliers(distance_matrix, contamination):
    # Compute LOF
    lof = LocalOutlierFactor(n_neighbors=20, metric='precomputed', contamination=contamination)
    lof_labels = lof.fit_predict(distance_matrix)
    
    # t-SNE for dimensionality reduction
    tsne = TSNE(metric='precomputed', init='random', n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(distance_matrix)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    colors = np.array(['blue' if label == 1 else 'red' for label in lof_labels])
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, s=10, alpha=0.5, edgecolor='k')
    
    # Remove axis values
    plt.xticks([])
    plt.yticks([])

    # Save and show the plot
    filename = f'TSNE_with_LOF_contamination_{contamination}.png'
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    # Assume all_trajectories is loaded or defined somewhere here
    all_trajectories = pd.read_pickle('formatted_trajectories.pkl')
    trajectory_keys = list(all_trajectories.keys())
    trajectory_data = [all_trajectories[key] for key in trajectory_keys]

    target_mmsi = 273418680
    start_date = datetime(2022, 1, 5)
    end_date = datetime(2022, 1, 9)

    # Prepare to filter trajectories
    mmsi_trajectories = {key: data for key, data in all_trajectories.items() if data['mmsi'] == target_mmsi}

    filtered_keys = []
    for key, data in mmsi_trajectories.items():
        timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if isinstance(ts, str) else ts for ts in data['timestamp']]
        if any(start_date <= ts <= end_date for ts in timestamps):
            filtered_keys.append(key)

    # Assuming you have filtered_keys and all_trajectories as dictionaries
    filtered_indices = [i for i, key in enumerate(trajectory_keys) if key in filtered_keys]
    print(filtered_indices)


    print(f"Number of trajectories for MMSI {target_mmsi} between {start_date} and {end_date}: {len(filtered_keys)}")

    all_trajectories = list(all_trajectories.values()) 
    

    all_traj = [simplify_trajectory(traj) for traj in all_trajectories]
    print(len(all_traj))

    if os.path.exists('distance_matrix.npy'):
        print("Loading existing distance matrix.")
        distance_matrix = load_distance_matrix()
    else:
        print("Computing new distance matrix.")
        distance_matrix = compute_or_load_distances(all_traj)

    # Continue processing with the distance matrix
    print("Distance matrix is ready for use.")
    print(distance_matrix)

    lof = LocalOutlierFactor(n_neighbors=20, metric='precomputed', contamination=0.34)
    lof_labels = lof.fit_predict(distance_matrix)

    outlier_statuses = [(index, lof_labels[index] == -1) for index in filtered_indices]
    for index, is_outlier in outlier_statuses:
        print(f"Trajectory at index {index} classified as {'outlier' if is_outlier else 'inlier'}")
    
    # t-SNE for dimensionality reduction
    tsne = TSNE(metric='precomputed', init='random', n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(distance_matrix)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    colors = np.array(['black' if label == 1 else 'plum' for label in lof_labels])

    colors = np.array(['black' if label == 1 else 'plum' for label in lof_labels])

    # Plot non-outliers first (assuming label != 1 are non-outliers)
    non_outliers = reduced_data[lof_labels != 1]
    plt.scatter(non_outliers[:, 0], non_outliers[:, 1], c='', s=10, alpha=0.5, edgecolor='k')

    # Plot outliers (assuming label == 1 are outliers)
    outliers = reduced_data[lof_labels == 1]
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red', s=10, alpha=0.5, edgecolor='none')

    
    highlight = np.array([index in filtered_indices for index in range(len(reduced_data))])
    plt.scatter(reduced_data[highlight, 0], reduced_data[highlight, 1], facecolors='none', edgecolors='yellow', s=20, linewidths=1.5)

    # Remove axis values
    plt.xticks([])
    plt.yticks([])

    # Save and show the plot
    filename = f'TSNE_with_LOF_contamination_34.png'
    plt.savefig(filename)
    plt.show()
