# -*- encoding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt


def eucl_distance(p1, p2):
    """
    计算欧式距离
    """
    return np.sqrt(np.sum(np.power(p1 - p2, 2)))


def init_centroids(data_set, k):
    """
    随机初始化k个中心点
    """
    num_samples, dim = data_set.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, num_samples))
        centroids[i, :] = data_set[index, :]
    return centroids


def kmeans(data_set, k):
    num_samples = data_set.shape[0]
    cluster_assment = np.mat(np.zeros((num_samples, 2)))
    cluster_changed = True
    # step 1: 初始化k个中心点
    centroids = init_centroids(data_set, k)
    while cluster_changed:
        cluster_changed = False
        for i in range(num_samples):
            min_dist = 100000.0
            min_index = 0
            # step 2: 计算最近距离的一个中心点
            for j in range(k):
                distance = eucl_distance(data_set[i, :], centroids[j, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            ## step 3: 更新族群点
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
                cluster_assment[i, :] = min_index, min_dist ** 2
            ## step 4: 更新中心点
            for j in range(k):
                points_in_cluster = data_set[np.nonzero(cluster_assment[:, 0].A == j)[0]]
                if len(points_in_cluster) > 0:
                    centroids[j, :] = np.mean(points_in_cluster, axis=0)
                else:
                    # 如果族群中没有点，则随机重新初始化该中心点
                    centroids[j, :] = init_centroids(data_set, 1)


    return centroids, cluster_assment


def show_cluster(data_set, k, centroids, cluster_assment):
    plt.title(u'Kmeans')
    num_samples, dim = data_set.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(num_samples):
        mark_index = int(cluster_assment[i, 0])
        plt.plot(data_set[i, 0], data_set[i, 1], mark[mark_index])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.xlim(0.0, 100)
    plt.ylim(0.0, 100)
    plt.show()


def init_data(num, min, max):
    data = []
    for i in range(num):
        data.append([np.random.randint(min, max), np.random.randint(min, max)])

    return data


if __name__ == "__main__":
    data_set1 = init_data(10, 0, 30)
    data_set2 = init_data(10, 30, 60)
    data_set3 = init_data(10, 60, 100)
    data_set = data_set1+data_set2+data_set3
    data_set = np.mat(data_set)
    k = 6
    centroids, cluster_assment = kmeans(data_set, k)
    show_cluster(data_set, k, centroids, cluster_assment)
print('finished')
