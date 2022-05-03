# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import itertools
import sys
import time
import random
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]


class DataLoader:
    def transfer(self, r):
        r[0] = int(r[0])
        r[1] = int(r[1])
        for i in range(2, len(r)):
            r[i] = float(r[i])
        return r

    def __init__(self):
        with open(input_file) as f:
            self.all_data = f.readlines()
        self.all_data = list(map(lambda r: r.strip('\n').split(','), self.all_data))
        self.all_data = list(map(lambda r: self.transfer(r), self.all_data))
        self.data_num = len(self.all_data) // 5

        points_id = list(map(lambda r: r[0], self.all_data))
        self.points_feature = list(map(lambda r: np.array(r[2:]), self.all_data))
        points_feature_tuple = list(map(lambda r: tuple(r[2:]), self.all_data))
        self.feature_id_dic = dict(zip(points_feature_tuple, points_id))
        random.shuffle(self.points_feature)

    def load_20_percent(self, i):
        if i==4: return self.points_feature[i*self.data_num:]
        else: return self.points_feature[i*self.data_num:(i+1)*self.data_num]


class BFR:
    def __init__(self):
        self.DS = defaultdict(dict)  # Discard set, format: {cluster_label: N, SUM, SUMSQ}
        self.CS = defaultdict(dict)  # Compression set, format: {cluster_label: N, SUM, SUMSQ}
        self.RS = []  # Retained set, format: point features

    def count_cluster_point_num(self):
        cluster_points_list = defaultdict(list)  # label: [point_position, ...]
        for point_id, label in enumerate(k_means.labels_):
            cluster_points_list[label].append(point_id)
        return cluster_points_list

    def init_RS(self):
        cluster_points_list = self.count_cluster_point_num()
        RS_cluster_points_id = []
        for label, points_id in cluster_points_list.items():
            if len(points_id) < 10:
                RS_cluster_points_id += points_id # add point id in label (list)
                for p_id in points_id: # add all point features
                    self.RS.append(data0[p_id])
        for id in reversed(sorted(RS_cluster_points_id)):
            data0.pop(id) # remove those clusters from origin data

    def generate_DS(self):
        for point_id, label in enumerate(k_means.labels_):
            point_feature = data0[point_id]
            if label not in self.DS:
                self.DS[label] = {}
                self.DS[label]['N'] = [dataLoader.feature_id_dic[tuple(point_feature)]]
                self.DS[label]['SUM'] = point_feature
                self.DS[label]["SUMSQ"] = point_feature ** 2
            else:
                self.DS[label]['N'].append(dataLoader.feature_id_dic[tuple(point_feature)])
                self.DS[label]['SUM'] += point_feature
                self.DS[label]["SUMSQ"] += point_feature ** 2

    def generate_CS(self):
        cluster_points_list = self.count_cluster_point_num()
        RS_cluster_points_id = []
        new_RS = []
        new_RS_id = []
        for label, points_id in cluster_points_list.items():
            if len(points_id) == 1:
                RS_cluster_points_id += points_id  # add point id in label (list)
                for p_id in points_id:  # add all point features
                    new_RS.append(self.RS[p_id])
                    new_RS_id.append(p_id)
        for point_id, label in enumerate(k_means.labels_):
            point_feature = self.RS[point_id]
            if point_id not in new_RS_id:
                if label not in self.CS:
                    self.CS[label] = {}
                    self.CS[label]['N'] = [dataLoader.feature_id_dic[tuple(point_feature)]]
                    self.CS[label]['SUM'] = point_feature
                    self.CS[label]["SUMSQ"] = point_feature ** 2
                else:
                    self.CS[label]['N'].append(dataLoader.feature_id_dic[tuple(point_feature)])
                    self.CS[label]['SUM'] += point_feature
                    self.CS[label]["SUMSQ"] += point_feature ** 2
        self.RS = new_RS

    def output_intermediate_results(self, round):
        # “the number of the discard points”
        DS_point_count = 0
        for label in self.DS:
            DS_point_count += len(self.DS[label]["N"])
        # “the number of the clusters in the compression set”
        CS_clusters_count = len(self.CS)
        # “the number of the compression points”
        CS_point_count = 0
        for label in self.CS:
            CS_point_count += len(self.CS[label]["N"])
        # “the number of the points in the retained set”.
        RS_point_count = len(self.RS)
        with open(output_file, 'a+') as f:
            f.writelines('Round '+str(round)+': '+str(DS_point_count)+','+str(CS_clusters_count)+','
                         +str(CS_point_count)+','+str(RS_point_count)+'\n')

    def find_nearest_cluster_and_distance(self, point, cluster_type):
        nearest_cluster = []
        nearest_dis = float('inf')
        cluster = self.DS if cluster_type == 'DS' else self.CS
        for k, v in cluster.items():
            centroid = v["SUM"] / len(v["N"])
            sigma = v["SUMSQ"] / len(v["N"]) - (v["SUM"] / len(v["N"])) ** 2
            y = (point - centroid) / sigma
            distance = sum(y ** 2) ** (1 / 2)
            if distance < nearest_dis:
                nearest_cluster = k
                nearest_dis = distance
        return nearest_cluster, nearest_dis

    def compute_cluster_distance(self, k1, k2, k2_type):
        if k2_type=='CS':
            cluster1, cluster2 = self.CS[k1], self.CS[k2]
        else:
            cluster1, cluster2 = self.CS[k1], self.DS[k2]
        centroid1 = cluster1["SUM"] / len(cluster1["N"])
        centroid2 = cluster2["SUM"] / len(cluster2["N"])
        sig_1 = cluster1["SUMSQ"] / len(cluster1["N"]) - (cluster1["SUM"] / len(cluster1["N"])) ** 2
        sig_2 = cluster2["SUMSQ"] / len(cluster2["N"]) - (cluster2["SUM"] / len(cluster2["N"])) ** 2
        y = centroid1 - centroid2
        y1, y2 = y/sig_1, y/sig_2
        distance = min(sum(y1 ** 2) ** (1 / 2), sum(y2 ** 2) ** (1 / 2))
        return distance

    def merge_CS(self):
        while True:
            pairs = list(itertools.combinations(list(self.CS.keys()), 2))
            prev_CS_cluster_num = len(self.CS)
            for p in pairs:
                dis = self.compute_cluster_distance(p[0], p[1], 'CS')
                d = len(self.CS[p[0]]["SUM"])
                if dis < 2 * (d ** (1 / 2)):
                    self.CS[p[0]]["N"] = self.CS[p[0]]["N"] + self.CS[p[1]]["N"]
                    self.CS[p[0]]["SUM"] += self.CS[p[1]]["SUM"]
                    self.CS[p[0]]["SUMSQ"] += self.CS[p[1]]["SUMSQ"]
                    self.CS.pop(p[1])
                    break
            if len(self.CS)==prev_CS_cluster_num:
                break

    def merge_CS_DS(self):
        CS_key_list = list(self.CS.keys())
        for CS_key in CS_key_list:
            min_dis = float('inf')
            min_DS_key = ''
            for DS_key in self.DS:
                dis = self.compute_cluster_distance(CS_key, DS_key, "DS")
                if dis < min_dis:
                    min_dis = dis
                    min_DS_key = DS_key
            if min_dis < 2 * len(self.CS[CS_key]["SUM"]) ** (1 / 2):
                self.DS[min_DS_key]["N"] = self.DS[min_DS_key]["N"] + self.CS[CS_key]["N"]
                self.DS[min_DS_key]["SUM"] += self.CS[CS_key]["SUM"]
                self.DS[min_DS_key]["SUMSQ"] += self.CS[CS_key]["SUMSQ"]
                self.CS.pop(CS_key)


if __name__ == '__main__':
    start_time = time.time()

    dataLoader = DataLoader()

    if 'init' == 'init': # first round
        # Step 1. Load 20% of the data randomly.
        data0 = dataLoader.load_20_percent(0)
        # print('data0', data0[0], ' , num data0', len(data0))
        # Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)
        k_means = KMeans(n_clusters=n_cluster * 5).fit(data0)
        print(k_means, 'labels: ', k_means.labels_)
        # Step 3. move all the clusters that contain only one point to RS
        bfr = BFR()
        bfr.init_RS()
        print('RS: ', len(bfr.RS), len(data0)) # check RS has been moved from data0
        # Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
        k_means = KMeans(n_clusters=n_cluster).fit(data0)
        # Step 5. Use the K-Means result from Step 4 to generate the DS clusters
        bfr.generate_DS()
        # Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters)
        # to generate CS (clusters with more than one points) and RS (clusters with only one point).
        if len(bfr.RS) > 1:
            cluster_num = min(n_cluster * 5, len(bfr.RS) - 1)
            # cluster_num = min(n_cluster * 5, len(bfr.RS) // 2)
            k_means = KMeans(n_clusters=cluster_num).fit(bfr.RS)
        else:
            k_means = KMeans(n_clusters=len(bfr.RS)).fit(bfr.RS)
        print('step6 kmeans', k_means)
        bfr.generate_CS()
        # write round 1 intermediate result
        with open(output_file, 'w+') as f:
            f.writelines('The intermediate results:\n')
        bfr.output_intermediate_results(1)

    # round 1 to 4
    for i in range(1, 5):
        # Step 7. Load another 20% of the data randomly.
        data_i = dataLoader.load_20_percent(i)
        # Step 8. For the new points, compare them to each of the DS using the Mahalanobis
        # Distance and assign them to the nearest DS clusters if the distance is < 2 𝑑.
        data_i_assigned_id = set()
        for point_id in range(len(data_i)):
            point = data_i[point_id]
            nearest_cluster, nearest_dis = bfr.find_nearest_cluster_and_distance(point, 'DS')
            if nearest_dis < 2 * (len(point) ** (1 / 2)):
                bfr.DS[nearest_cluster]['N'].append(dataLoader.feature_id_dic[tuple(point)])
                bfr.DS[nearest_cluster]["SUM"] += point
                bfr.DS[nearest_cluster]["SUMSQ"] += point ** 2
                data_i_assigned_id.add(point_id)
        print('step ', i, 'data assigned to DS', len(data_i_assigned_id))
        # Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis
        # Distance and assign the points to the nearest CS clusters if the distance is < 2 𝑑
        for point_id in range(len(data_i)):
            if point_id not in data_i_assigned_id:
                point = data_i[point_id]
                nearest_cluster, nearest_dis = bfr.find_nearest_cluster_and_distance(point, 'CS')
                if nearest_dis < 2 * (len(point) ** (1 / 2)):
                    bfr.CS[nearest_cluster]['N'].append(dataLoader.feature_id_dic[tuple(point)])
                    bfr.CS[nearest_cluster]["SUM"] += point
                    bfr.CS[nearest_cluster]["SUMSQ"] += point ** 2
                    data_i_assigned_id.add(point_id)
        # Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
        for point_id in range(len(data_i)):
            if point_id not in data_i_assigned_id:
                point = data_i[point_id]
                bfr.RS.append(point)
        # Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to
        # generate CS (clusters with more than one points) and RS (clusters with only one point).
        if len(bfr.RS) > 1:
            cluster_num = min(n_cluster * 5, len(bfr.RS) - 1)
            # cluster_num = min(n_cluster * 5, len(bfr.RS) // 2)
            k_means = KMeans(n_clusters=cluster_num).fit(bfr.RS)
        else:
            k_means = KMeans(n_clusters=len(bfr.RS)).fit(bfr.RS)
        bfr.generate_CS()
        # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2 𝑑.
        bfr.merge_CS()
        if i == 4:
            bfr.merge_CS_DS()
        bfr.output_intermediate_results(i + 1)

    res = '\n' + 'The clustering results:' + '\n'
    for cluster in bfr.DS:
        bfr.DS[cluster]["N"] = set(bfr.DS[cluster]["N"])
    if bfr.CS:
        for cluster in bfr.CS:
            bfr.CS[cluster]["N"] = set(bfr.CS[cluster]["N"])

    RS_id_set = set()
    for RS_feature in bfr.RS:
        RS_id_set.add(dataLoader.feature_id_dic[tuple(RS_feature)])
    for point_id in range(len(dataLoader.all_data)):
        if point_id in RS_id_set:
            res += str(point_id) + ",-1\n"
        else:
            for cluster in bfr.DS:
                if point_id in bfr.DS[cluster]["N"]:
                    res += str(point_id) + "," + str(cluster) + "\n"
                    break
            for cluster in bfr.CS:
                if point_id in bfr.CS[cluster]["N"]:
                    res += str(point_id) + ",-1\n"
                    break

    with open(output_file, 'a+') as f:
        f.writelines(res)

    print('Duration: ', time.time() - start_time)
