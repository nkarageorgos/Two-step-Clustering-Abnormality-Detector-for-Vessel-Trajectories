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



def plot_compressed_trajectories(original, simplified, epsilon, index):
    plt.figure(figsize=(10, 6))
    plt.plot(original['lon'], original['lat'], 'b-', label='Original')
    plt.plot(simplified['lon'], simplified['lat'], 'r-', label=f'Compressed with epsilon={epsilon})')
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'Compressions{index}')
    plt.show()

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



# Define the Haversine function to calculate distances between two points
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Function to calculate the average Haversine distance according to the formula
def average_haversine(traj1, traj2):
    # Assuming traj1 and traj2 are dictionaries with 'lon' and 'lat' as lists of coordinates
    min_length = min(len(traj1['lon']), len(traj2['lon'])) - 1
    sum_distances = 0

    for i in range(min_length):
        sum_distances += haversine(traj1['lon'][i], traj1['lat'][i], traj2['lon'][i], traj2['lat'][i])
        sum_distances += haversine(traj1['lon'][i+1], traj1['lat'][i+1], traj2['lon'][i+1], traj2['lat'][i+1])

    # The final average includes dividing the summed trapezoidal approximations by 2T
    return sum_distances / (2 * (min_length + 1))
# Function to compute distances for a single trajectory against all others
def calculate_haversine_distances(index, all_trajectories):
    dists = [average_haversine(all_trajectories[index], traj) if i >= index else 0 for i, traj in enumerate(all_trajectories)]
    return dists

def directed_hausdorff_spherical(u, v):
    max_min_dist = 0
    for point1 in u:
        min_dist = np.inf
        for point2 in v:
            dist = haversine(point1[1], point1[0], point2[1], point2[0])
            if dist < min_dist:
                min_dist = dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist
    return max_min_dist

def calculate_hausdorff_distances(index, all_trajectories):
    traj1 = all_trajectories[index]
    dists = [0] * len(all_trajectories)  # Initialize distances with zeros
    for i, traj2 in enumerate(all_trajectories):
        if i >= index:
            forward = directed_hausdorff_spherical(np.array(traj1), np.array(traj2))
            backward = directed_hausdorff_spherical(np.array(traj2), np.array(traj1))
            dists[i] = max(forward, backward)
    return dists


def circle_distance(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371.0  # Earth radius in kilometers
    return R * c

def compute_dtw(t0, t1):
    n0, n1 = len(t0), len(t1)
    C = np.full((n0+1, n1+1), np.inf)
    C[0, 0] = 0
    for i in range(1, n0+1):
        for j in range(1, n1+1):
            dist = circle_distance(t0[i-1][0], t0[i-1][1], t1[j-1][0], t1[j-1][1])
            C[i, j] = dist + min(C[i-1, j], C[i, j-1], C[i-1, j-1])
    return C[n0, n1]

def calculate_dtw_distances(index, all_trajectories):
    traj1 = all_trajectories[index]
    return [compute_dtw(traj1, traj2) if i >= index else 0 for i, traj2 in enumerate(all_trajectories)]


def d_speed(x, y, sigma):
    if sigma == 0:
        return 0
    return np.abs(x - y) / (sigma + np.finfo(float).eps) 

def d_course(x, y):
    """Calculate the normalized angular difference in radians."""
    angular_difference = abs(x - y)
    if angular_difference <= np.pi:
        return angular_difference / np.pi
    else:
        return (2 * np.pi - angular_difference) / np.pi

def compute_kindtw_distance(s1, s2, distance_function):
    """Compute DTW for given sequences using specified distance function."""
    n, m = len(s1), len(s2)
    DTW = np.full((n + 1, m + 1), float('inf'))
    DTW[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = distance_function(s1[i - 1], s2[j - 1])
            DTW[i, j] = cost + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    return DTW[n, m]

def calculate_dkin_distances(trajectory1, trajectory2):
    """Calculate the kinematic DTW distance between two trajectories."""
    speeds = np.concatenate([trajectory1[:, 1], trajectory2[:, 1]])  # Standard deviation of speed
    sigma = np.std(speeds)
    
    # Check for NaNs in speeds
    if np.isnan(speeds).any():
        print("NaN values found in speed data.")


    dtw_speed = compute_kindtw_distance(trajectory1[:, 1], trajectory2[:, 1], lambda x, y: d_speed(x, y, sigma))
    dtw_course = compute_kindtw_distance(trajectory1[:, 0], trajectory2[:, 0], d_course)
    return np.abs(dtw_speed + dtw_course)

def worker(index, trajectories, n):
    """Calculate distances for the upper triangle of the matrix from the given index."""
    distances = []
    for j in range(index + 1, n):
        distance = calculate_dkin_distances(trajectories[index], trajectories[j])
        distances.append((index, j, distance))
    return distances

def calculate_kinematic_distance_matrix(trajectories, num_processes=16):
    """Calculate the kinematic DTW distance matrix using parallel processing."""
    n = len(trajectories)
    tasks = [(i, trajectories, n) for i in range(n)]

    with Pool(num_processes) as pool:
        results = pool.starmap(worker, tasks)

    distance_matrix = np.zeros((n, n))

    for result in results:
        for i, j, distance in result:
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Since the matrix is symmetric

    return distance_matrix


# Parallel computation of distance matrix
def parallel_distance_matrix(all_trajectories, num_processes):
    with Pool(processes=num_processes) as pool:
        result = pool.starmap(calculate_haversine_distances, [(i, all_trajectories) for i in range(len(all_trajectories))])
    # Fill in the lower triangular part of the distance matrix
    n = len(all_trajectories)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                distance_matrix[i][j] = result[i][j]
            else:
                distance_matrix[i][j] = distance_matrix[j][i]
    return distance_matrix



# Save the distance matrix to a .npy file
def save_distance_matrix(matrix, filename):
    np.save(filename, matrix)

# Load the distance matrix from a .npy file
def load_distance_matrix(filename):
    return np.load(filename, allow_pickle=False)

# Main function to compute or load the distance matrix
def compute_or_load_distances(all_trajectories, filename='haversinematrix.npy', num_processes=16):
    if (os.path.exists(filename)):
        print("Loading existing distance matrix.")
        return load_distance_matrix(filename)
    else:
        print("Computing distance matrix.")
        matrix = parallel_distance_matrix(all_trajectories, num_processes)
        save_distance_matrix(matrix, filename)
        return matrix
    
def compute_or_load_kindistances(all_trajectories, filename='dkinematicmatrix.npy', num_processes=16):
    if (os.path.exists(filename)):
        print("Loading existing distance matrix.")
        return load_distance_matrix(filename)
    else:
        print("Computing distance matrix.")
        matrix = parallel_distance_matrix(all_trajectories, num_processes)
        save_distance_matrix(matrix, filename)
        return matrix

def plot_k_distance(distance_matrix, k):
    nn = NearestNeighbors(n_neighbors=k,metric='precomputed')
    nn.fit(distance_matrix)
    distances, _ = nn.kneighbors(distance_matrix)
    
    # Sort the distance to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])[::-1]

    plt.figure(figsize=(8, 4))
    plt.plot(k_distances)
    plt.title(f"k-Distance Graph for k={k}")
    plt.xlabel('Points')
    plt.ylabel('k-th Nearest Distance')
    plt.show()

    # Use kneed to find the knee point, which suggests a good eps
    knee_locator = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='decreasing')
    return knee_locator.knee_y 

def plot_speedcourse_trajectories(trajectories):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Limit the number of trajectories to plot
    limited_trajectories = trajectories[:20]

    # Speed vs. Time (Duration)
    for traj in limited_trajectories:
        # Create an array from 0 to length of the speed array - 1
        duration = np.arange(len(traj['speed']))
        axes[0].plot(duration, traj['speed'], linewidth=0.5)  # Set linewidth to 0.5 for thinner lines
    axes[0].set_title('Speed vs. Duration')
    axes[0].set_xlabel('Duration (index)')
    axes[0].set_ylabel('Speed (units)')

    # Course vs. Time (Duration)
    for traj in limited_trajectories:
        # Create an array from 0 to length of the course array - 1
        duration = np.arange(len(traj['course']))
        axes[1].plot(duration, traj['course'], linewidth=0.5)  # Set linewidth to 0.5 for thinner lines
    axes[1].set_title('Course vs. Duration')
    axes[1].set_xlabel('Duration (index)')
    axes[1].set_ylabel('Course (degrees)')

    plt.tight_layout()
    plt.savefig('SpeedCourseCluster2')
    plt.show()


def plot_trajectories_with_map(trajectories, labels, main_cluster, singletons_indices):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx

    # Prepare a GeoDataFrame
    gdf_list = []
    for traj in trajectories:
        gdf = gpd.GeoDataFrame(traj, geometry=gpd.points_from_xy(traj['lon'], traj['lat']))
        gdf.set_crs(epsg=4326, inplace=True)
        gdf_list.append(gdf)
    
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot trajectories of the main cluster
    main_cluster_trajectories = [gdf_list[i] for i in range(len(labels)) if labels[i] == main_cluster]
    for gdf in main_cluster_trajectories:
        gdf = gdf.to_crs(epsg=3857)  # Convert to the same CRS as the basemap
        ax.plot(gdf.geometry.x, gdf.geometry.y, color='#9f9595', alpha=0.7)  
    
    for singleton_idx in singletons_indices:
        gdf = gdf_list[singleton_idx].to_crs(epsg=3857)  
        ax.plot(gdf.geometry.x, gdf.geometry.y, color='#f31111', alpha=0.7)  # Lighter purple
    
    # Set the plot limits
    x_min, x_max = 880631.7937419682, 1910042.086929147
    y_min, y_max = 13742466.61716103, 15199337.050095655
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron)

    # Hide axis labels and ticks
    ax.set_ylabel('')  
    ax.set_xlabel('')  
    ax.set_xticks([])  
    ax.set_yticks([])  
    ax.set_title(f'Cluster 7')

    plt.savefig('Clusters')
    plt.show()

def apply_lof(X, contamination=0.1):
    lof = LocalOutlierFactor(n_neighbors=20, metric='precomputed', contamination=contamination)
    # Fit the LOF and predict outlier labels
    labels = lof.fit_predict(distance_kinmatrix)
    return labels

def plot_tsne_results(X_tsne, labels, highlight_indices, filename):
    plt.figure(figsize=(5, 4))
    colors = {1: 'blue', -1: 'red'}  # 1: inlier, -1: outlier
    
    # Scatter plot adjustments
    scatter_plots = {}
    for idx, (x, y) in enumerate(X_tsne):
        label = 'Outlier' if labels[idx] == -1 else 'Inlier'
        if label not in scatter_plots:
            scatter_plots[label] = plt.scatter([], [], color=colors[labels[idx]], label=label)
        plt.scatter(x, y, color=colors[labels[idx]], s=10) 
        if idx in highlight_indices:
            circle = plt.Circle((x, y), radius=0.06, color='black', fill=False, linewidth=1)
            plt.gca().add_patch(circle)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    for label, plot in scatter_plots.items():
        handles.append(plot)
    
    plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xticks([])
    plt.yticks([])

    plt.savefig(filename, bbox_inches='tight') 
    plt.close()


def plot_trajectories_with_map_separate(trajectories, labels, cluster_indices, cluster_colors):
    gdf_list = []
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
    
    for traj in trajectories:
        gdf = gpd.GeoDataFrame(traj, geometry=gpd.points_from_xy(traj['lon'], traj['lat']))
        gdf.set_crs(epsg=4326, inplace=True)
        gdf = gdf.to_crs(epsg=3857)  # Convert to the same CRS as the basemap
        gdf_list.append(gdf)
        
        # Update the overall min and max coordinates
        x_min = min(x_min, gdf.geometry.x.min())
        x_max = max(x_max, gdf.geometry.x.max())
        y_min = min(y_min, gdf.geometry.y.min())
        y_max = max(y_max, gdf.geometry.y.max())
    
    # Number of rows and columns for the subplots
    rows, cols = 2, 4  

    # Create the figure with subplots
    fig, axs = plt.subplots(rows, cols, figsize=(17, 10))  
    axs = axs.flatten() 

    # Plot trajectories of the clusters
    for idx, (cluster_idx, color) in enumerate(sorted(zip(cluster_indices, cluster_colors))):
        ax = axs[idx] 
        cluster_trajectories = [gdf_list[i] for i in range(len(labels)) if labels[i] == cluster_idx]
        for gdf in cluster_trajectories:
            ax.plot(gdf.geometry.x, gdf.geometry.y, color=color, alpha=0.5)
        
        # Set the same limits for each subplot to ensure uniform size and scale
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        # Add the basemap
        ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron)

        # Remove axis labels and values
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Cluster {cluster_idx+1}')

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig('Adjusted_Clusters_Trajectories.png')
    plt.show()

def summarize_clustering(labels):
    # Count the number of unique labels; -1 represents outliers
    unique_labels = np.unique(labels)
    n_clusters = len([label for label in unique_labels if label != -1])
    n_outliers = np.count_nonzero(labels == -1)
    n_total_points = len(labels)
    outlier_percentage = (n_outliers / n_total_points) * 100
    
    # Count the number of points per cluster excluding outliers
    cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
    median_cluster_size = np.median(cluster_sizes) if cluster_sizes else 0
    
    # Create a summary dictionary
    summary = {
        'Number of clusters': n_clusters,
        'Number of outliers': n_outliers,
        'Percentage of outliers': outlier_percentage,
        'Median cluster size': median_cluster_size
    }
    
    return summary

def get_original_trajectories_by_cluster(labels, original_trajectories):
    clustered_originals = {}
    for label in set(labels):
        if label != -1:  # Exclude outliers if you wish
            indices = [i for i, cluster_label in enumerate(labels) if cluster_label == label]
            clustered_originals[label] = [original_trajectories[i] for i in indices]
    return clustered_originals


def convert_to_initial_format(clustered_trajectories):
    # Dictionary to hold the converted format
    formatted_trajectories_by_cluster = {}
    
    # Iterate through each cluster
    for cluster_label, trajectories in clustered_trajectories.items():
        # This assumes each trajectory in 'trajectories' is a dictionary format like the original input
        formatted_trajectories_by_cluster[cluster_label] = trajectories  # Directly assign list of trajectory dicts
    
    return formatted_trajectories_by_cluster

def plot_speedcourse_similartrajectories(trajectories, pairs):
    # Calculate the total number of unique trajectories in the pairs
    unique_traj_indices = set()
    for pair in pairs:
        unique_traj_indices.update(pair)
    unique_trajectories = [trajectories[idx] for idx in unique_traj_indices]

    # Create plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Speed vs. Time (Duration)
    for traj in unique_trajectories:
        duration = np.arange(len(traj['speed']))
        axes[0].plot(duration, traj['speed'], linewidth=0.5)  # Set linewidth to 0.5 for thinner lines
    axes[0].set_title('Speed vs. Duration')
    axes[0].set_xlabel('Duration (index)')
    axes[0].set_ylabel('Speed (units)')

    # Course vs. Time (Duration)
    for traj in unique_trajectories:
        duration = np.arange(len(traj['course']))
        axes[1].plot(duration, traj['course'], linewidth=0.5)  # Set linewidth to 0.5 for thinner lines
    axes[1].set_title('Course vs. Duration')
    axes[1].set_xlabel('Duration (index)')
    axes[1].set_ylabel('Course (degrees)')

    plt.tight_layout()
    plt.savefig('SpeedCourseCluster6')
    plt.show()

def simplify_trajectory(trajectory, epsilon=0.0005):
    # This function takes a dataframe or a dictionary with 'lon' and 'lat' keys and an epsilon value for simplification.
    coords = np.array(list(zip(trajectory['lon'], trajectory['lat'])))
    simplified_coords = simplify_coords(coords, epsilon)
    return {'lon': simplified_coords[:, 0], 'lat': simplified_coords[:, 1]}

def simplify_trajectory2(trajectory, speed_threshold=0.15, course_threshold=1.5):
    
    simplified_trajectory = {
        'lon': [trajectory['lon'][0]],
        'lat': [trajectory['lat'][0]],
        'speed': [trajectory['speed'][0]],
        'course': [trajectory['course'][0]]
    }

    for i in range(1, len(trajectory['lon']) - 1):
        # Calculate changes in speed and course
        speed_change = abs(trajectory['speed'][i] - trajectory['speed'][i - 1])
        course_change = abs(trajectory['course'][i] - trajectory['course'][i - 1])
        
        # Adjust for course wrapping around 360 degrees
        if course_change > 180:
            course_change = 360 - course_change
        
        # Check if the change exceeds thresholds
        if speed_change > speed_threshold or course_change > course_threshold:
            simplified_trajectory['lon'].append(trajectory['lon'][i])
            simplified_trajectory['lat'].append(trajectory['lat'][i])
            simplified_trajectory['speed'].append(trajectory['speed'][i])
            simplified_trajectory['course'].append(trajectory['course'][i])

    # Always include the last point
    last_index = len(trajectory['lon']) - 1
    simplified_trajectory['lon'].append(trajectory['lon'][last_index])
    simplified_trajectory['lat'].append(trajectory['lat'][last_index])
    simplified_trajectory['speed'].append(trajectory['speed'][last_index])
    simplified_trajectory['course'].append(trajectory['course'][last_index])

    return simplified_trajectory


def plot_trajectories_with_map_cluster6(all_trajectories, labels, filtered_keys, special_clusters):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import contextily as ctx

    # Prepare a GeoDataFrame
    gdf_list = []
    for key, traj in all_trajectories.items():
        gdf = gpd.GeoDataFrame(traj, geometry=gpd.points_from_xy(traj['lon'], traj['lat']))
        gdf.set_crs(epsg=4326, inplace=True)
        gdf['key'] = key  # store the key for filtering
        gdf_list.append(gdf)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all trajectories
    for gdf in gdf_list:
        gdf_projected = gdf.to_crs(epsg=3857)
        ax.plot(gdf_projected.geometry.x, gdf_projected.geometry.y, color='#9f9595', alpha=0.5)  # general trajectories in grey

    # Highlight trajectories in the specified timeframe
    filtered_gdfs = [gdf for gdf in gdf_list if gdf['key'][0] in filtered_keys]
    for gdf in filtered_gdfs:
        gdf_projected = gdf.to_crs(epsg=3857)
        ax.plot(gdf_projected.geometry.x, gdf_projected.geometry.y, color='blue', alpha=0.7)  # filtered trajectories in blue

    # Highlight special trajectories (non-zero cluster labels)
    special_keys = [key for key, cluster in labels.items() if cluster in special_clusters and cluster != 0]
    special_gdfs = [gdf for gdf in gdf_list if gdf['key'][0] in special_keys]
    for gdf in special_gdfs:
        gdf_projected = gdf.to_crs(epsg=3857)
        ax.plot(gdf_projected.geometry.x, gdf_projected.geometry.y, color='red', alpha=0.9)  # special trajectories in red

    print("Number of trajectories to plot in red:", len(special_gdfs))
    for gdf in special_gdfs:
        print("Preview of trajectory data:", gdf.head())

    # Set the plot limits and add the basemap
    ax.set_xlim([880631.7937419682, 1910042.086929147])
    ax.set_ylim([13742466.61716103, 15199337.050095655])
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron)

    # Hide axis labels and ticks
    ax.set_ylabel('')  # Removes y-axis label
    ax.set_xlabel('')  # Removes x-axis label
    ax.set_xticks([])  # Removes x-axis ticks
    ax.set_yticks([])  # Removes y-axis ticks
    

    plt.savefig('Clusters_with_special_cases')
    plt.show()

def save_trajectories_to_excel(trajectories, filename='cluster_6_trajectories.xlsx'):
    # Convert each trajectory's data into a DataFrame, assuming each key in the dictionary is a column
    dfs = []
    for idx, traj in enumerate(trajectories):
        df = pd.DataFrame(traj)
        df['trajectory_id'] = f'traj_{idx}'  # Add a trajectory identifier column
        dfs.append(df)

        # Concatenate all dataframes into a single dataframe
    all_data = pd.concat(dfs, ignore_index=True)

        # Save to Excel
    all_data.to_excel(filename, index=False)

def find_most_similar_trajectories(distance_matrix, num_results=5):
    # We will ignore the diagonal by setting it to infinity
    np.fill_diagonal(distance_matrix, np.inf)
    
    # Flatten the matrix and get the indices of the smallest values
    indices = np.argsort(distance_matrix, axis=None)
    
    # Convert 1D indices to 2D indices
    x, y = np.unravel_index(indices, distance_matrix.shape)
    
    # We need unique pairs, excluding self-comparisons
    unique_pairs = []
    seen = set()
    for i, j in zip(x, y):
        if i != j and i not in seen and j not in seen:
            unique_pairs.append((i, j))
            seen.update([i, j])
            if len(unique_pairs) == num_results:
                break
    
    return unique_pairs

def plot_selected_speedsourse(trajectories, singleton_index, other_indices):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    # Plot the singleton with a unique color
    duration_singleton_speed = np.arange(len(trajectories[singleton_index]['speed']))
    duration_singleton_course = np.arange(len(trajectories[singleton_index]['course']))
    axes[0].plot(duration_singleton_speed, trajectories[singleton_index]['speed'], color='red')
    axes[1].plot(duration_singleton_course, trajectories[singleton_index]['course'], color='red')

    # Plot each similar trajectory with a different color
    for idx in other_indices:
        duration_speed = np.arange(len(trajectories[idx]['speed']))
        duration_course = np.arange(len(trajectories[idx]['course']))
        axes[0].plot(duration_speed, trajectories[idx]['speed'], color='blue')
        axes[1].plot(duration_course, trajectories[idx]['course'], color='blue')

    axes[0].set_title('Speed vs Duration')
    axes[0].set_xlabel('Duration')
    axes[0].set_ylabel('Speed')

    axes[1].set_title('Course vs Duration')
    axes[1].set_xlabel('Duration')
    axes[1].set_ylabel('Course')

    plt.tight_layout()
    plt.savefig('cluster6')
    plt.show()



if __name__ == "__main__":

    all_trajectories = pd.read_pickle('formatted_trajectories.pkl')  # adjust path as necessary
    trajectory_keys = list(all_trajectories.keys())
    trajectory_data = [all_trajectories[key] for key in trajectory_keys]

    target_mmsi = 273418680
    start_date = datetime(2022, 1, 5)
    end_date = datetime(2022, 1, 9)

    mmsi_trajectories = {key: data for key, data in all_trajectories.items() if data['mmsi'] == target_mmsi}

    # Further filter trajectories within the specified timeframe
    filtered_keys = []
    for key, data in mmsi_trajectories.items():
        timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if isinstance(ts, str) else ts for ts in data['timestamp']]
        if any(start_date <= ts <= end_date for ts in timestamps):
            filtered_keys.append(key)

    print(f"Number of trajectories for MMSI {target_mmsi} between {start_date} and {end_date}: {len(filtered_keys)}")
    all_trajectories = list(all_trajectories.values())  # convert dictionary values to a list if necessary

    #this is for the hausdorff+dtw calc
    #all_trajectories = [np.array(list(zip(traj['lon'], traj['lat']))) for traj in all_trajectories]

    original_trajectories = [traj.copy() for traj in all_trajectories]

    all_traj_array = [simplify_trajectory(traj) for traj in all_trajectories]

    # Convert simplified trajectories to numpy arrays for distance calculations
    #all_traj_array = [np.array(list(zip(traj['lon'], traj['lat']))) for traj in simplified_trajectories]

    print("Number of keys:", len(trajectory_keys))
    print("Number of processed trajectories:", len(all_traj_array))


    # Compute or load the distance matrix
    num_processes = 16  
    distance_matrix = compute_or_load_distances(all_traj_array, num_processes=num_processes)
    print(distance_matrix)
   

    linkage_methods = ['average']
    for method in linkage_methods:
        plt.figure(figsize=(10, 7))
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, metric='precomputed', linkage=method)
        model.fit(distance_matrix)
        plot_dendrogram(model, truncate_mode='level', p=5)
        plt.title(f'Dendrogram for {method} linkage')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.savefig(f'Dendogram {method}')
        plt.show()

    # Identify the optimal number of clusters using silhouette score
    best_score = -1

    clustering = AgglomerativeClustering(n_clusters=8, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(distance_matrix)
    if len(set(labels)) > 1:  # Valid cluster formation check
        score = silhouette_score(distance_matrix, labels, metric='precomputed')

    print(f"Silhouette Score: {score}")

    cluster_counts = Counter(labels)
    print("Number of trajectories in each cluster:", cluster_counts)

        
    clustering_summary = summarize_clustering(labels)
    print(clustering_summary)

    model = AgglomerativeClustering(n_clusters=8, linkage='average', compute_full_tree=True, metric='precomputed')
    model.fit(distance_matrix)
    labels = model.labels_



    if len(labels) == len(trajectory_keys):
        trajectory_labels = {trajectory_keys[i]: labels[i] for i in range(len(trajectory_keys))}
        print("Trajectory labels mapped successfully.")
    else:
        print("Error: The number of labels does not match the number of trajectory keys.")

    cluster_counts = Counter(labels)
    print("Number of trajectories in each cluster:", cluster_counts)

    cluster_6_keys = [key for key, label in trajectory_labels.items() if label == 6]

    # Filter to get only those keys that are in the filtered keys list and in Cluster 6
    filtered_cluster_6_keys = [key for key in filtered_keys if key in cluster_6_keys]

    print("Filtered Trajectory Keys in Cluster 6:", filtered_cluster_6_keys)

    # Map cluster labels back to original trajectory keys
    trajectory_labels = {trajectory_keys[i]: labels[i] for i in range(len(trajectory_keys))}

    # Extract labels for filtered trajectories
    filtered_trajectory_labels = {key: trajectory_labels[key] for key in filtered_keys}
    print("Cluster labels for filtered trajectories:", filtered_trajectory_labels)

    # Optional: Analyze the clustering result

        
    clustering_summary = summarize_clustering(labels)
    print(clustering_summary)

    clustered_original_trajectories = get_original_trajectories_by_cluster(labels, original_trajectories)

    formatted_trajectories_by_cluster = convert_to_initial_format(clustered_original_trajectories)

    cluster_0_trajectories = formatted_trajectories_by_cluster.get(0, [])
    cluster_1_trajectories = formatted_trajectories_by_cluster.get(1, [])
    cluster_2_trajectories = formatted_trajectories_by_cluster.get(2, [])
    cluster_3_trajectories = formatted_trajectories_by_cluster.get(3, [])
    cluster_4_trajectories = formatted_trajectories_by_cluster.get(4, [])
    cluster_5_trajectories = formatted_trajectories_by_cluster.get(5, [])
    cluster_6_trajectories = formatted_trajectories_by_cluster.get(6, [])
    cluster_7_trajectories = formatted_trajectories_by_cluster.get(7, [])
    
    
    
    simplified_0_trajectories = [simplify_trajectory2(traj) for traj in cluster_0_trajectories]
    simplified_0_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_0_trajectories]
    simplified_1_trajectories = [simplify_trajectory2(traj) for traj in cluster_1_trajectories]
    simplified_1_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_1_trajectories]
    simplified_2_trajectories = [simplify_trajectory2(traj) for traj in cluster_2_trajectories]
    simplified_2_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_2_trajectories]
    simplified_3_trajectories = [simplify_trajectory2(traj) for traj in cluster_3_trajectories]
    simplified_3_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_3_trajectories]
    simplified_4_trajectories = [simplify_trajectory2(traj) for traj in cluster_4_trajectories]
    simplified_4_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_4_trajectories]
    simplified_5_trajectories = [simplify_trajectory2(traj) for traj in cluster_5_trajectories]
    simplified_5_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_5_trajectories]
    simplified_6_trajectories = [simplify_trajectory2(traj) for traj in cluster_6_trajectories]
    simplified_6_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_6_trajectories]
    simplified_7_trajectories = [simplify_trajectory2(traj) for traj in cluster_7_trajectories]
    simplified_7_trajectories = [np.array(list(zip(traj['course'], traj['speed']))) for traj in simplified_7_trajectories]


    distance_kinmatrix = compute_or_load_kindistances(simplified_6_trajectories, num_processes=num_processes)
    distance_kinmatrix = np.abs(distance_kinmatrix)
    distance_kinmatrixcopy = distance_kinmatrix
    print(distance_kinmatrix)

    all_distances = distance_kinmatrix[np.triu_indices_from(distance_kinmatrix, k=1)]


    mean_distance = np.mean(all_distances)
    median_distance = np.median(all_distances)
    std_deviation = np.std(all_distances)
    percentile90 = np.percentile(all_distances, 74)

    print("Mean distance:", mean_distance/100)
    print("Median distance:", median_distance/100)
    print("Standard deviation:", std_deviation/100)
    print("Percentile 90:", percentile90/100)

    # Perform clustering with the best parameters
    best_clustering = AgglomerativeClustering(distance_threshold=percentile90, n_clusters=None, linkage='average', metric='precomputed')
    labels2 = best_clustering.fit_predict(distance_kinmatrix)

    def find_singletons(labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        singleton_labels = unique_labels[counts == 1]
        singleton_indices = [i for i, label in enumerate(labels) if label in singleton_labels]
        return singleton_indices, singleton_labels

    # Get singletons
    singleton_indices, singleton_labels = find_singletons(labels2)

    score = silhouette_score(distance_kinmatrix, labels2, metric='precomputed')
    print(f"Silhouette Score: {score}")

    clustering_summary = summarize_clustering(labels2)
    print(clustering_summary)
    # Print results
    print("Singleton Labels:", singleton_labels)
    print("Number of Singletons:", len(singleton_indices))
    filtered_trajectory_labels = {filtered_keys[i]: labels2[i] for i in range(len(filtered_keys))}
    print("New labels for filtered trajectories in Cluster 6:", filtered_trajectory_labels)

    # Find the most populous clusters
    unique, counts = np.unique(labels2, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    # Sort clusters by size (optional, for reference)
    sorted_clusters = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)

    # Get the main cluster label (the most populous one)
    main_cluster = sorted_clusters[0][0]

    special_clusters = [16, 12, 22]

    similar_pairs = find_most_similar_trajectories(distance_kinmatrix)
    print("Index pairs of the most similar trajectories:", similar_pairs)

    plot_trajectories_with_map_cluster6(mmsi_trajectories, filtered_trajectory_labels , filtered_keys, special_clusters)
    plot_speedcourse_trajectories(cluster_1_trajectories)

    plot_speedcourse_similartrajectories(cluster_6_trajectories, similar_pairs)

    #save_trajectories_to_excel(cluster_6_trajectories)

    # Plot trajectories of the main cluster and singletons
    #plot_trajectories_with_map(cluster_6_trajectories, labels2, main_cluster, singleton_indices)

    singleton_index = np.where(labels2 == 16)[0][0] 
    valid_indices = [i for i in np.where(labels2 == 0)[0] if len(cluster_6_trajectories[i]['speed']) > 300]
    valid_distances = distance_kinmatrix[singleton_index][valid_indices]

    closest_valid_indices = np.argsort(valid_distances)[:5]
    closest_global_indices = [valid_indices[i] for i in closest_valid_indices]
    
    #plot_selected_trajectoriesC(cluster_6_trajectories, np.append([singleton_index], closest_global_indices))
    plot_selected_speedsourse(cluster_6_trajectories, singleton_index, closest_global_indices)

    if np.isinf(distance_kinmatrixcopy).any():
        max_finite = np.nanmax(np.where(np.isfinite(distance_kinmatrixcopy), distance_kinmatrixcopy, -np.inf))
        distance_kinmatrixcopy[np.isinf(distance_kinmatrixcopy)] = max_finite + 1e5 

    if np.isnan(distance_kinmatrixcopy).any():
        distance_kinmatrixcopy[np.isnan(distance_kinmatrixcopy)] = max_finite + 1e5

    np.fill_diagonal(distance_kinmatrixcopy, 0)  
    distance_kinmatrixcopy[np.isinf(distance_kinmatrixcopy)] = np.max(distance_kinmatrixcopy[np.isfinite(distance_kinmatrixcopy)])  # Replace infinities

 # Initialize t-SNE with the "random" init parameter since "pca" cannot be used with "precomputed"
    tsne = TSNE(n_components=2, perplexity=30, metric='precomputed', init='random')
    X_tsne = tsne.fit_transform(distance_kinmatrixcopy)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
    plt.title('t-SNE visualization of precomputed distances')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('TNSE')
    plt.show()

    keys = [
        'MMSI_273418680_Traj_92', 'MMSI_273418680_Traj_93', 'MMSI_273418680_Traj_94',
        'MMSI_273418680_Traj_95', 'MMSI_273418680_Traj_96', 'MMSI_273418680_Traj_97',
        'MMSI_273418680_Traj_98', 'MMSI_273418680_Traj_99', 'MMSI_273418680_Traj_100'
    ]

    # Find indices of these keys in cluster_6_keys
    highlight_indices = [index for index, key in enumerate(cluster_6_keys) if key in keys]

    contamination_levels = [0.01, 0.05, 0.09, 0.15]
    for contam in contamination_levels:
        outlier_labels = apply_lof(X_tsne, contamination=contam)
        filename = f"t-SNE_LOF_Contam_{contam:.2f}.png"  # Format filename to include contamination level
        plot_title = f"t-SNE with LOF Outliers (Contam. {contam:.2f})"
        plot_tsne_results(X_tsne, outlier_labels, highlight_indices, filename=filename)
         # Count and print the number of outliers
        num_outliers = (outlier_labels == -1).sum()
        print(f"Number of outliers at contamination level {contam:.2f}: {num_outliers}")

    
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    sorted_clusters = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)
    top_clusters = [cluster[0] for cluster in sorted_clusters[:8]]  # Get top 8 clusters
    top_clusters_colors = sns.color_palette("hsv", len(top_clusters))  # Get different colors for the clusters

    plot_trajectories_with_map_separate(all_trajectories, labels, top_clusters, top_clusters_colors)
    