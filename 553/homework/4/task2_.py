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
    edges = set()
    vertexes = set()
    for pair in candidate_pairs:
        business1 = user_business_dict[pair[0]]
        business2 = user_business_dict[pair[1]]
        if len(business1 & business2) >= int(filter_threshold):
            vertexes.add(pair[0])
            vertexes.add(pair[1])
            edges.add(tuple(pair))
            edges.add(tuple((pair[1], pair[0])))
    edges = sc.parallelize(edges).groupByKey().mapValues(lambda r: sorted(list(r))).collectAsMap()
    return edges, list(vertexes)



class MyGraph():
    def __init__(self, edges, vertexes):
        self.edges = edges
        self.vertexes = vertexes
        # {edge: betweenness}
        self.betweenness = collections.defaultdict(float)
        self.betweenness_rlt = []
        self.count_betweennesses()
        # pre calculate for Q
        self.edge_num = 0
        for k in self.edges.keys():
            self.edge_num += len(self.edges[k])
        # self.adjacent_matrix = edges
        # build adjacent matrix for original edges
        edge_set = set()
        for start_node, end_nodes in edges.items():
            for end_node in end_nodes:
                pair = (start_node, end_node) if start_node < end_node else (end_node, start_node)
                edge_set.add(pair)
        self.A_matrix = edge_set

    def bfs_count_shortest_path_num(self, root):
        self.parent_set = {}
        self.parent_set[root] = (0, []) # (node_level, node parents)
        next_nodes = queue.Queue()
        next_nodes.put(root)
        visited = set()
        while next_nodes.empty() == False:
            curr_node = next_nodes.get()
            visited.add(curr_node)
            for next in self.edges[curr_node]:
                if next not in visited:
                    visited.add(next)
                    next_nodes.put(next)
                    curr_level = self.parent_set[curr_node][0]
                    self.parent_set[next] = (curr_level+1, [curr_node])
                # 如果next被访问过，且curr_node来自next的上一层。说明有两条边接入next。将curr node加入next的parent列表
                else:
                    next_level, curr_level = self.parent_set[next][0], self.parent_set[curr_node][0]
                    if next_level-1==curr_level:
                        self.parent_set[next][1].append(curr_node)

        level_dict = collections.defaultdict(list)
        for child_node, (level, parents) in self.parent_set.items():
            level_dict[level].append((child_node, parents))

        self.shortest_path_num = collections.defaultdict(int)
        for level in range(0, len(level_dict.keys())):
            for (child_node, parents_list) in level_dict[level]:
                if level == 0: # root node the shortest path number set to 1
                    self.shortest_path_num[child_node] = 1
                else:
                    for p in parents_list:
                        self.shortest_path_num[child_node] += self.shortest_path_num[p]

    def traverse_and_count_betweenness(self):
        self.vertex_weight_dict = collections.defaultdict(float)
        [self.vertex_weight_dict.setdefault(vertex, 1) for vertex in self.vertexes]
        # sort by node level
        self.parent_set = {k: v for k, v in sorted(self.parent_set.items(), key=lambda kv: -kv[1][0])}
        for child, parents in self.parent_set.items():
            for parent in parents[1]:
                contribution = self.vertex_weight_dict[child] * self.shortest_path_num[parent] / self.shortest_path_num[child]
                curr_edge = (child, parent) if child < parent else (parent, child)
                self.betweenness[curr_edge] += contribution
                self.vertex_weight_dict[parent] += contribution

    def count_betweennesses(self):
        self.betweenness = collections.defaultdict(float)
        self.betweenness_rlt = []
        for i in range(len(self.vertexes)):
            node = self.vertexes[i]
            self.bfs_count_shortest_path_num(node) # visit each node and compute the number of the shortest path
            self.traverse_and_count_betweenness()

        for k in self.betweenness.keys():
            self.betweenness[k] /= 2
            if k[0] < k[1]:
                self.betweenness_rlt.append((float(self.betweenness[k]), k))
        self.betweenness_rlt.sort(key=lambda r: (-r[0], r[1]))

    def remove_edges(self):
        max_betweenness = self.betweenness_rlt[0][0]
        # print(self.betweenness_rlt[:3])
        for item in self.betweenness_rlt:
            if item[0] >= max_betweenness:
                edge = item[1]
                # print('removing this edge: ', edge)
                # print('edges storage detail: ',len(self.edges[edge[0]]), len(self.edges[edge[1]]))
                if self.edges[edge[0]] is not None:
                    try:
                        self.edges[edge[0]].remove(edge[1])
                    except ValueError:
                        pass
                if self.edges[edge[1]] is not None:
                    try:
                        self.edges[edge[1]].remove(edge[0])
                    except ValueError:
                        pass
                # print('edges storage detail after remove: ', len(self.edges[edge[0]]), len(self.edges[edge[1]]))
            else: break

    def count_modularity(self):
        sum = 0
        for cluster in self.curr_communities:
            for pair in combinations(list(cluster), 2):
                pair = pair if pair[0] < pair[1] else (pair[1], pair[0])
                k0 = len(self.edges[pair[0]])
                k1 = len(self.edges[pair[1]])
                # A_ij = 1 if (pair[1] in self.adjacent_matrix[pair[0]]) else 0
                A_ij = 1 if pair in self.A_matrix else 0
                sum += float(A_ij - ( k0*k1 / self.edge_num ) )
        return float(sum / self.edge_num)

    def search_current_communities(self):
        self.curr_communities = []
        # use a set to record if a node is visited globally
        global_visited = set()
        # use one_community to record the community we are exploring
        one_community = set()
        next_visit = [self.vertexes[0]]

        # do bfs search, until all vertexes are visited and put into one community:
        while len(global_visited) < len(self.vertexes):
            while len(next_visit) > 0:
                curr_node = next_visit.pop(0)
                one_community.add(curr_node)
                global_visited.add(curr_node)
                for next_node in self.edges[curr_node]:
                    if next_node not in global_visited:
                        next_visit.append(next_node)
            self.curr_communities.append(sorted(one_community))
            one_community = set() # clear community for next loop
            if len(global_visited) < len(self.vertexes):
                next_root = set(self.vertexes).difference(global_visited).pop()
                next_visit.append(next_root)
        return self.curr_communities

    def find_best_communities(self):
        max_modularity = -1
        self.best_community = self.search_current_communities()
        curr_modularity = self.count_modularity()

        while True:
            # print('\n')
            # print('remove_log', curr_modularity, self.betweenness_rlt[0])
            self.remove_edges()
            curr_community = self.search_current_communities()
            curr_modularity = self.count_modularity()
            self.count_betweennesses()
            if curr_modularity > max_modularity:
                self.best_community = curr_community
                max_modularity = curr_modularity
            else:
                break
        self.best_community.sort(key = lambda r: (len(r), r))
        return self.best_community


if __name__ == "__main__":
    start_time = time.time()
    sc = SparkContext()
    sparkSession = SparkSession(sc)
    sc.setLogLevel("WARN")

    data = sc.textFile(input_file_path, 20)
    header = data.first()
    data = data.filter(lambda r: r != header).map(lambda r: (r.split(',')[0], r.split(',')[1]))

    user_business = data.groupByKey().mapValues(set)
    user_set = user_business.map(lambda r: r[0]).collect()
    user_business_dict = user_business.collectAsMap()

    candidate_pairs = list(combinations(user_set, 2))
    edges, vertexes = filter_pairs()
    # print('test edges: ', len(edges))
    # print('test vertexes: ', len(vertexes))

    my_graph = MyGraph(edges, vertexes)
    # print(my_graph.shortest_path_num)
    # print('test shortest_path_num: ', len(my_graph.shortest_path_num))
    # print('test betweenness: ', len(my_graph.betweenness))

    with open(betweenness_output_file_path, 'w') as f:
        for r in my_graph.betweenness_rlt:
            f.writelines(str(r[1])+', '+str(round(r[0], 5))+'\n')

    # init_community = my_graph.search_current_communities()
    # print('init communities len: ', len(init_community)) # for test
    best_community = my_graph.find_best_communities()
    with open(community_output_file_path, 'w') as f:
        for r in best_community:
            f.writelines(str(r)[1:-1] + "\n")

    print('Duration: ', time.time() - start_time)