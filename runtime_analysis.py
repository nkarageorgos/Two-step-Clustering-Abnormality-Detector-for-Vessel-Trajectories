import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from multiprocessing import Pool
import os
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from fastdtw import fastdtw
from simplification.cutil import simplify_coords
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from collections import Counter
import time

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
def calculate_avghaversine_distances(index, all_trajectories):
    traj1 = all_trajectories[index]
    dists = [0] * len(all_trajectories)
    times = []
    for i, traj2 in enumerate(all_trajectories):
        if i >= index:
            start_time = time.time()
            dists[i] = average_haversine(traj1, traj2)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
    return dists, times

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
    dists = [0] * len(all_trajectories)
    times = []
    for i, traj2 in enumerate(all_trajectories):
        if i >= index:
            start_time = time.time()
            forward = directed_hausdorff_spherical(np.array(traj1), np.array(traj2))
            backward = directed_hausdorff_spherical(np.array(traj2), np.array(traj1))
            times.append(time.time() - start_time)
            dists[i] = max(forward, backward)
    return dists, times

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

def parallel_distance_matrix(all_trajectories, num_processes):
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(calculate_avghaversine_distances, [(i, all_trajectories) for i in range(len(all_trajectories))])
    n = len(all_trajectories)
    distance_matrix = np.zeros((n, n))
    total_times = []
    for i, (dists, times) in enumerate(results):
        total_times.extend(times)
        for j in range(n):
            if j >= i:
                distance_matrix[i][j] = dists[j]
            else:
                distance_matrix[i][j] = distance_matrix[j][i]
    return distance_matrix, total_times


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
    dists = [0] * len(all_trajectories)
    times = []
    for i, traj2 in enumerate(all_trajectories):
        if i >= index:
            start_time = time.time()
            dists[i] = compute_dtw(traj1, traj2)
            times.append(time.time() - start_time)
    return dists, times

if __name__ == "__main__":
    all_trajectories = pd.read_pickle('formatted_trajectories.pkl')
    all_trajectories = list(all_trajectories.values())  

    all_traj = [simplify_trajectory(traj) for traj in all_trajectories]

    # Convert simplified trajectories to numpy arrays for distance calculations
    #all_traj= [np.array(list(zip(traj['lon'], traj['lat']))) for traj in simplified_trajectories]

    all_traj = all_traj[:1000]


      # Load only 1000 trajectories
    num_processes = 16  # Adjust based on your server's capability
    distance_matrix, computation_times = parallel_distance_matrix(all_traj, num_processes)
    average_time_per_pair = np.mean(computation_times)
    print(f"Average computation time per pair: {average_time_per_pair} seconds")
    print(distance_matrix)
