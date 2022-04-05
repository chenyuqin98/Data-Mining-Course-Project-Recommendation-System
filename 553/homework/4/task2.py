import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from itertools import combinations
import time
import collections
import queue
from tqdm import tqdm


filter_threshold = sys.argv[1]
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]


def filter_pairs():
    edges = list()
    vertexes = set()
    for pair in candidate_pairs:
        business1 = user_business_dict[pair[0]]
        business2 = user_business_dict[pair[1]]
        if len(business1 & business2) >= int(filter_threshold):
            vertexes.add(pair[0])
            vertexes.add(pair[1])
            edges.append(tuple(pair))
            edges.append(tuple((pair[1], pair[0])))
    return edges, list(vertexes)



class MyGraph():
    def __init__(self, edges, vertexes):
        self.edges = edges
        self.vertexes = vertexes
        # {edge: betweenness}
        self.betweenness = collections.defaultdict(float)
        self.count_betweennesses()

    def bfs_count_shortest_path_num(self, root):
        # {vertex: shortest_path_num from root to vertex}
        self.shortest_path_num = collections.defaultdict(int)
        self.shortest_path_num[root] = 1
        # {{
        self.bfs_tree = collections.defaultdict(set)

        next_nodes = queue.Queue()
        next_nodes.put(('dummy', root, 0))
        level_dic = collections.defaultdict(set)

        def judge(e, curr_node, prev_node):
            if e[0] == curr_node:
                if e[1] == prev_node:
                    return False
                for level in range(curr_level, -1, -1):
                    if e[1] in level_dic[level]:
                        return False
                return True
            return False

        while next_nodes.empty() == False:
            prev_node, curr_node, curr_level = next_nodes.get()
            if prev_node != 'dummy':
                self.bfs_tree[prev_node].add(curr_node)
            self.shortest_path_num[curr_node] += self.shortest_path_num[prev_node]
            # print('prev_node:', prev_node, ' curr_node:', curr_node, ' curr_level:', curr_level, ' parent num:', self.shortest_path_num[curr_node])
            for e in self.edges:
                if judge(e, curr_node, prev_node):
                    next_nodes.put((e[0], e[1], curr_level + 1))
                    level_dic[curr_level + 1].add(e[1])
        # print('bfs_tree', self.bfs_tree)

    def dfs_count_betweenness(self, node, curr_edge=None):
        node_val = 1 # 每个node的初始weight
        # update node from bottom to top
        for next_node in self.bfs_tree[node]:
            node_val += self.dfs_count_betweenness(next_node,(node, next_node))

        if curr_edge is not None:
            contribute = float(float(node_val) * int(self.shortest_path_num[curr_edge[0]]) / self.shortest_path_num[node])
            if curr_edge[0] > curr_edge[1]:
                curr_edge = (curr_edge[1], curr_edge[0])
            self.betweenness[curr_edge] += contribute
            # if curr_edge == ('0FMte0z-repSVWSJ_BaQTg', '0FVcoJko1kfZCrJRfssfIA'): # test
            if curr_edge[0] == 'cyuDrrG5eEK-TZI867MUPA':
                print('189 contribute test:', node_val, self.shortest_path_num[curr_edge[0]], self.shortest_path_num[node])
            return contribute
        return 0 # root

    def count_betweennesses(self):
        print('------------', '0FMte0z-repSVWSJ_BaQTg')
        self.bfs_count_shortest_path_num('0FMte0z-repSVWSJ_BaQTg')
        self.dfs_count_betweenness('0FMte0z-repSVWSJ_BaQTg')
        # for i in range(len(self.vertexes)):
        #     node = self.vertexes[i]
        #     self.bfs_count_shortest_path_num(node) # visit each node and compute the number of the shortest path
        #     self.dfs_count_betweenness(node)
        for k in self.betweenness.keys():
            self.betweenness[k] /= 2


if __name__ == "__main__":
    start_time = time.time()
    sc = SparkContext()
    sparkSession = SparkSession(sc)
    sc.setLogLevel("WARN")

    data = sc.textFile(input_file_path, 20)
    header = data.first()
    data = data.filter(lambda r: r != header).map(lambda r: (r.split(',')[0], r.split(',')[1]))
    # print(data.first())

    user_business = data.groupByKey().mapValues(set)
    user_set = user_business.map(lambda r: r[0]).collect()
    user_business_dict = user_business.collectAsMap()
    # print(user_set[0])

    candidate_pairs = list(combinations(user_set, 2))
    # print(candidate_pairs[0], len(candidate_pairs))

    edges, vertexes = filter_pairs()
    print('test edges: ', edges[0], len(edges))
    print('test vertexes: ', vertexes[0], len(vertexes))
    # for test
    # with open('edge_vertex_test.txt', 'w') as f:
    #     f.writelines(str(edges)+'\n')
    #     f.writelines(str(vertexes)+'\n')

    my_graph = MyGraph(edges, vertexes)
    # print(my_graph.shortest_path_num)
    print('test shortest_path_num: ', len(my_graph.shortest_path_num))
    print('test betweenness: ', len(my_graph.betweenness))

    betweenness_rlt = []
    for k in my_graph.betweenness.keys():
        if k[0] < k[1]:
            betweenness_rlt.append((float(my_graph.betweenness[k]), k))
    betweenness_rlt.sort(key=lambda r:(-r[0], r[1]))
    print(betweenness_rlt[:5])
    with open(betweenness_output_file_path, 'w') as f:
        for r in betweenness_rlt:
            f.writelines(str(r[1])+', '+str(round(r[0], 5))+'\n')

    print('Duration: ', time.time() - start_time)