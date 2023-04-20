from collections import defaultdict
import numpy as np
from itertools import product
from scipy.special import gamma
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform, euclidean
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from random import randint


def volume(r, m):
    return np.pi ** (m / 2) * r ** m / gamma(m / 2 + 1)


def significant(cluster, h, p):
    max_diff = max(abs(p[i] - p[j]) for i, j in product(cluster, cluster))

    return max_diff >= h


def partition(dist, l, r, order):
    if l == r:
        return l

    pivot = dist[order[(l + r) // 2]]
    left, right = l - 1, r + 1
    while True:
        while True:
            left += 1
            if dist[order[left]] >= pivot:
                break

        while True:
            right -= 1
            if dist[order[right]] <= pivot:
                break

        if left >= right:
            return right

        order[left], order[right] = order[right], order[left]
        


def detrend(data_matrix):
    ans = []
    for data_list in data_matrix:
        series = pd.Series(data_list)
        X = [i for i in range(0, len(series))]
        X = np.reshape(X, (len(X), 1))
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        detrended = [y[i]-trend[i] for i in range(0, len(series))]
        ans.append(detrended)
    return ans


def detrend_flat(data_list):
    series = pd.Series(data_list)
    X = [i for i in range(0, len(series))]
    X = np.reshape(X, (len(X), 1))
    y = series.values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    detrended = [y[i]-trend[i] for i in range(0, len(series))]
    return detrended

        
def nth_element(dist, order, k):
    l = 0
    r = len(order) - 1
    while True:
        if l == r:
            break
        m = partition(dist, l, r, order)
        if m < k:
            l = m + 1
        elif m >= k:
            r = m

            
def get_clustering(x, k, h, verbose=True):
    n = len(x)
    if isinstance(x[0], list):
        m = len(x[0])
    else:
        m = 1
    dist = squareform(pdist(x)) #checkpoint №1

    dk = []
    for i in range(n):
        order = list(range(n))
        nth_element(dist[i], order, k - 1)
        dk.append(dist[i][order[k - 1]])

    p = [k / (volume(dk[i], m) * n) for i in range(n)]

    w = np.full(n, 0)
    completed = {0: False}
    last = 1
    vertices = set()
    for d, i in sorted(zip(dk, range(n))):
        neigh = set()
        neigh_w = set()
        clusters = defaultdict(list)
        for j in vertices:
            if dist[i][j] <= dk[i]:
                neigh.add(j)
                neigh_w.add(w[j])
                clusters[w[j]].append(j)

        vertices.add(i)
        if len(neigh) == 0:
            w[i] = last
            completed[last] = False
            last += 1
        elif len(neigh_w) == 1:
            wj = next(iter(neigh_w))
            if completed[wj]:
                w[i] = 0
            else:
                w[i] = wj
        else:
            if all(completed[wj] for wj in neigh_w):
                w[i] = 0
                continue
            significant_clusters = set(wj for wj in neigh_w if significant(clusters[wj], h, p))
            if len(significant_clusters) > 1:
                w[i] = 0
                for wj in neigh_w:
                    if wj in significant_clusters:
                        completed[wj] = (wj != 0)
                    else:
                        for j in clusters[wj]:
                            w[j] = 0
            else:
                if len(significant_clusters) == 0:
                    s = next(iter(neigh_w))
                else:
                    s = next(iter(significant_clusters))
                w[i] = s
                for wj in neigh_w:
                    for j in clusters[wj]:
                        w[j] = s
    return w


def index_element(arr, i, partition, identifier):
    if (identifier == "max"):
        return i+arr[i:i+partition-1].index(max(arr[i:i+partition-1]))
    else:
        return i+arr[i:i+partition-1].index(min(arr[i:i+partition-1]))
    

#TODO: поделить ряд на n // 2 участке и на каждом участке брать min и max 
def generate_z_vector_best(arr, n):
    z_vector = []
    partition = (len(arr) * 2) // n
    for i in range(0,len(arr),partition):
        z_vector.append(index_element(arr, i, partition, "max"))
        z_vector.append(index_element(arr, i, partition, "min"))
    return z_vector
#я тут немного пошаманил и сделал дополнительную функцию чтобы наш говнокод выглядел хоть чуть-чуть получше

def generate_motif(clustered_data):
    motif = []
    transposed_data = np.transpose(clustered_data)
    for point in transposed_data:
        motif.append(np.average(point))
    return motif

def find_boundaries(visualization_data, vdf_min, vdf_max, time_stamp):
    #distance2 = math.hypot(max(motif), min(motif))
    #distance3 = abs(max(motif))*math.sqrt(1 + (min(motif)/max(motif))**2)
    scale = np.arange(vdf_min, vdf_max, (vdf_max-vdf_min)/100) # divide the y-axis into 100 "cells"

    count_on_scale = [0]*len(scale)
    for sri in range(len(scale)-1): # sri = scale_range_index, here we insert in every cell the number of series passing through it
        for time_series in visualization_data:
            if scale[sri] < time_series[time_stamp] < scale[sri + 1]:
                count_on_scale[sri] += 1

    for i in range(len(count_on_scale)):
        if count_on_scale[i] <= 0.65*max(count_on_scale):
            count_on_scale[i] = 0

    boundaries = []
    index_max = count_on_scale.index(max(count_on_scale))
    for i in range(index_max, 0, -1):
        if count_on_scale[i] == 0:
            boundaries.append(scale[i])
            break
    for i in range(index_max, len(count_on_scale)):
        if count_on_scale[i] == 0:
            boundaries.append(scale[i])
            break
    return boundaries


class Wishart:
    clusters_to_objects: defaultdict
    object_labels: np.ndarray
    clusters: np.ndarray
    kd_tree: cKDTree

    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors
        self.significance_level = significance_level

    def fit(self, X, workers=-1, batch_weight_in_gb=10):
        self.kd_tree = cKDTree(data=X)

        distances = np.empty(0).ravel()

        batch_size = batch_weight_in_gb * (1024 ** 3) // 8
        batches_count = X.shape[0] // (batch_size // (self.wishart_neighbors + 1))
        if batches_count == 0:
            batches_count = 1

        batches = np.array_split(X, batches_count)
        for batch in batches:
            batch_dists, _ = self.kd_tree.query(x=batch, k=self.wishart_neighbors + 1, n_jobs=workers) 
            # Changed in version 1.6.0: The “n_jobs” argument was renamed “workers”. The old name “n_jobs” is deprecated and will stop 
            # working in SciPy 1.8.0
            batch_dists = batch_dists[:, -1].ravel()
            distances = np.hstack((distances, batch_dists))

        indexes = np.argsort(distances)
        X = X[indexes]

        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype=int) - 1

        # index in tuple
        # min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)

        batches = np.array_split(X, batches_count)
        idx_batches = np.array_split(indexes, batches_count)
        del X, indexes

        for batch, idx_batch in zip(batches, idx_batches):
            _, neighbors = self.kd_tree.query(x=batch, k=self.wishart_neighbors + 1, n_jobs=workers)
            neighbors = neighbors[:, 1:]

            for real_index, idx in enumerate(idx_batch):
                neighbors_clusters = np.concatenate(
                    [self.object_labels[neighbors[real_index]], self.object_labels[neighbors[real_index]]])
                unique_clusters = np.unique(neighbors_clusters).astype(int)
                unique_clusters = unique_clusters[unique_clusters != -1]

                if len(unique_clusters) == 0:
                    self._create_new_cluster(idx, distances[idx])
                else:
                    max_cluster = unique_clusters[-1]
                    min_cluster = unique_clusters[0]
                    if max_cluster == min_cluster:
                        if self.clusters[max_cluster][-1] < 0.5:
                            self._add_elem_to_exist_cluster(idx, distances[idx], max_cluster)
                        else:
                            self._add_elem_to_noise(idx)
                    else:
                        my_clusters = self.clusters[unique_clusters]
                        flags = my_clusters[:, -1]
                        if np.min(flags) > 0.5:
                            self._add_elem_to_noise(idx)
                        else:
                            significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                            significan *= self.wishart_neighbors
                            significan /= size
                            significan /= np.power(np.pi, dim / 2)
                            significan *= gamma(dim / 2 + 1)
                            significan_index = significan >= self.significance_level

                            significan_clusters = unique_clusters[significan_index]
                            not_significan_clusters = unique_clusters[~significan_index]
                            significan_clusters_count = len(significan_clusters)
                            if significan_clusters_count > 1 or min_cluster == 0:
                                self._add_elem_to_noise(idx)
                                self.clusters[significan_clusters, -1] = 1
                                for not_sig_cluster in not_significan_clusters:
                                    if not_sig_cluster == 0:
                                        continue

                                    for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                        self._add_elem_to_noise(bad_index)
                                    self.clusters_to_objects[not_sig_cluster].clear()
                            else:
                                for cur_cluster in unique_clusters:
                                    if cur_cluster == min_cluster:
                                        continue

                                    for bad_index in self.clusters_to_objects[cur_cluster]:
                                        self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                    self.clusters_to_objects[cur_cluster].clear()

                                self._add_elem_to_exist_cluster(idx, distances[idx], min_cluster)

        return self.clean_data()

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq: index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype=int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis=0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)
