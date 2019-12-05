import numpy as np
from random import random
import math
from collections import defaultdict


def random_point(dmin, dmax):
    '''
    Get a random point given the bounds of the data
    :param dmin: numpy array for minimum bounds
    :param dmax: numpy array for maximum bounds
    :return:
    '''
    datapoint = []
    for i in range(dmin.shape[0]):
        rand_value = dmin[i] + (random() * (dmax[i] - dmin[i]))
        datapoint.append(rand_value)
    return np.array(datapoint)

class LSH:
    def __init__(self, layers, num_hashes, bins):
        self.N_LAYERS = layers
        self.N_BINS = bins
        self.N_HASHES = num_hashes
        self.layers = []
        self.hashes = {}
        self.hash_meta = {}
        self.buckets = defaultdict(lambda: defaultdict(list))
        for i in range(layers):
            self.layers.append([])
        pass

    @staticmethod
    def _random_point(dmin, dmax):
        '''
        Get a random point given the bounds of the data
        :param dmin: numpy array for minimum bounds
        :param dmax: numpy array for maximum bounds
        :return:
        '''
        datapoint = []
        for i in range(dmin.shape[0]):
            rand_value = dmin[i] + (random() * (dmax[i] - dmin[i]))
            datapoint.append(rand_value)
        return np.array(datapoint)

    def get_similar(self, image_vector, k):
        # num_neighbors = 0
        adjacent_bucket = 0
        neighbors = []
        while len(neighbors) < k+1:
            for layer_id, hash_names in enumerate(self.layers):
                layer_hash_bucket = []
                for i, hash_name in enumerate(hash_names):
                    start = self.hashes[hash_name][0]
                    end = self.hashes[hash_name][1]
                    datum_dot = np.dot((image_vector - start), (end - start))
                    datum_dot = abs((datum_dot - self.hash_meta[hash_name]['min'])
                                    * (self.N_BINS / self.hash_meta[hash_name]['range']))
                    bucket = int(math.floor(datum_dot))
                    if adjacent_bucket > 0:
                        n_bins = range(bucket-adjacent_bucket, bucket+adjacent_bucket+1)
                        adj_items = []
                        for b_i in n_bins:
                            adj_items = np.union1d(adj_items, self.buckets[hash_name][b_i])
                        if i == 0:
                            layer_hash_bucket = adj_items
                        else:
                            layer_hash_bucket = np.intersect1d(layer_hash_bucket, adj_items)
                    else:
                        if i == 0:
                            layer_hash_bucket = self.buckets[hash_name][bucket]
                        layer_hash_bucket = np.intersect1d(layer_hash_bucket, self.buckets[hash_name][bucket])
                neighbors = np.union1d(neighbors, layer_hash_bucket)
            adjacent_bucket += 1
        return [int(i) for i in neighbors]

    def fit(self, data):
        '''
        :param data: The data to fit
        :return:
        '''
        if len(data) == 0:
            print("LSH: Data with zero length provided")
            return

        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        # num_columns = data.shape[1]
        for i in range(self.N_LAYERS):
            for j in range(self.N_HASHES):
                hash_name = "h%i%i" % (i, j)
                point1 = self._random_point(data_min, data_max)
                point2 = self._random_point(data_min, data_max)
                self.layers[i].append(hash_name)
                self.hashes[hash_name] = (point1, point2)

        # Store each datapoint in a bucket for each hash function
        for k, v in self.hashes.items():
            start = v[0]
            end = v[1]
            dot_max = np.dot((data_max - start), (end - start))
            dot_min = np.dot((data_min - start), (end - start))

            dot_cache = {}
            for i, datum in enumerate(data):
                datum_dot = np.dot((datum - start), (end - start))
                if datum_dot > dot_max:
                    dot_max = datum_dot
                if datum_dot < dot_min:
                    dot_min = datum_dot

                dot_cache[i] = datum_dot

            dot_range = dot_max - dot_min
            self.hash_meta[k] = {
                'min': dot_min,
                'max': dot_max,
                'range': dot_range
            }
            for dat_k, dat_v in dot_cache.items():
                datum_dot = abs((dat_v - dot_min) * (self.N_BINS/dot_range))
                bucket = int(math.floor(datum_dot))
                if bucket > self.N_BINS:
                    print('LSH: out of index bucket')
                self.buckets[k][bucket].append(dat_k)
            del dot_cache


if __name__ == "__main__":
    print('Testing LSH')
    lsh = LSH(2, 3, 10)
    points = [random_point(np.array([-10, -10]), np.array([10, 10])) for x in list(range(30))]
    for i in range(10):
        points.append([0, 0])

    for point in points:
        print(point)

    lsh.fit(points)
    similar_indices = lsh.get_similar([0, 0], 3)
    print([list(points[i]) for i in similar_indices])

