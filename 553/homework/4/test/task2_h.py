import itertools
from itertools import combinations
import sys
import time
import random

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


def cmp(key1, key2):
    """
    :param key1:
    :param key2:
    :return:
    """
    return (key1, key2) if key1 < key2 else (key2, key1)


def export2File(result_array, file_path):
    """
    export list content to a file
    :param result_array: a list of dict
    :param file_path: output file path
    :return: nothing, but a file
    """
    with open(file_path, 'w+') as output_file:
        for id_array in result_array:
            output_file.writelines(str(id_array)[1:-1] + "\n")
        output_file.close()


def update_dict(dict_obj, key, increment):
    """
    update the value with the same key, rather than replace it
    :param dict_obj:
    :param key:
    :param increment:
    :return:
    """
    old_weight = dict_obj[key]
    dict_obj[key] = float(old_weight + increment)
    return dict_obj


def extend_dict(dict_obj, increment_dict):
    """
    same as list extend
    :param dict_obj:
    :param increment_dict:
    :return:
    """
    for key, value in increment_dict.items():
        if key in dict_obj.keys():
            dict_obj = update_dict(dict_obj, key, value)
        else:
            dict_obj[key] = value
    return dict_obj


class GraphFrame(object):

    def __init__(self, vertexes, edges):
        """
        :param vertexes: list of vertexes [1,2,3,4...]
        :param edges: a big dict(vertex: (list of vertex it connected)
        """
        self.vertexes = vertexes
        self.vertex_weight_dict = dict()
        self.__init_weight_dict__()

        self.edges = edges
        self.__init_adjacent_matrix__(edges)

        # variable using for compute betweenness
        self.betweenness_result_dict = dict()
        self.betweenness_result_tuple_list = None

        # variable using for compute modularity
        self.best_communities = None

        # pre calculate for Q
        self.edge_num = 0
        for k in self.edges.keys():
            self.edge_num += len(self.edges[k])
        self.adjacent_matrix = edges

    def __init_weight_dict__(self):
        [self.vertex_weight_dict.setdefault(vertex, 1) for vertex in self.vertexes]

    def __init_adjacent_matrix__(self, edges):
        """
        build a set which contain all edge pair
        :param edges: original edges (a big dict (vertex: [list of vertex it connected]))
        :return:
        """
        self.original_edges = edges
        self.m = self._count_edges(edges)

        # build adjacent matrix for original edges
        edge_set = set()
        for start_node, end_nodes in edges.items():
            for end_node in end_nodes:
                edge_set.add(cmp(start_node, end_node))
        self.A_matrix = edge_set

    def _count_edges(self, edges):
        """
        :param edges:  a big dict(vertex: (list of vertex it connected)
        :return:
        """
        visited = set()
        count = 0
        for start_node, end_nodes in edges.items():
            for end_node in end_nodes:
                key = cmp(start_node, end_node)
                if key not in visited:
                    visited.add(key)
                    count += 1
        return count

    def _build_tree(self, root):
        # root set in level 0 and no parent
        tree = dict()
        tree[root] = (0, list())

        # since BFS only visit each node once,
        # so use visited variable to save these records
        visited = set()

        need2visit = list()
        need2visit.append(root)

        while len(need2visit) > 0:
            parent_node = need2visit.pop(0)
            visited.add(parent_node)
            for children in self.edges[parent_node]:
                if children not in visited:
                    visited.add(children)
                    tree[children] = (tree[parent_node][0] + 1, [parent_node])
                    need2visit.append(children)
                elif tree[parent_node][0] + 1 == tree[children][0]:
                    tree[children][1].append(parent_node)

        return {k: v for k, v in sorted(tree.items(), key=lambda kv: -kv[1][0])}

    def _traverse_tree(self, tree_dict):
        """
        traverse the tree and compute weight for each edge
        :param tree_dict: {'2GUjO7NU88cPXpoffYCU8w': (9, ['a48HhwcmjFLApZhiax41IA']), ...
        :return:
        """
        weight_dict = self.vertex_weight_dict.copy()
        shortest_path_dict = self._find_num_of_paths(tree_dict)
        result_dict = dict()
        for key, value in tree_dict.items():
            if len(value[1]) > 0:
                denominator = sum([shortest_path_dict[parent] for parent in value[1]])
                for parent in value[1]:
                    temp_key = cmp(key, parent)
                    contribution = float(float(weight_dict[key]) * int(shortest_path_dict[parent]) / denominator)
                    # print('189 contribute test:', weight_dict[key], shortest_path_dict[parent], denominator)
                    # if temp_key == ('0FMte0z-repSVWSJ_BaQTg', '0FVcoJko1kfZCrJRfssfIA'):
                    #     print('189 contribute test:', weight_dict[key], shortest_path_dict[parent], denominator)
                    #     print(len(shortest_path_dict))
                    result_dict[temp_key] = contribution
                    # update every parent node weight
                    weight_dict = update_dict(weight_dict, parent, contribution)

        return result_dict

    def _find_num_of_paths(self, tree_dict):
        """
        find how many the number of shortest path each node has
        :param tree_dict: {'2GUjO7NU88cPXpoffYCU8w': (9, ['a48HhwcmjFLApZhiax41IA']), ...
        :return: {'y6jsaAXFstAJkf53R4_y4Q': 1, '0FVcoJko1kfZCrJRfssfIA': 1, '2quguRdKBzul ...
        """
        level_dict = dict()
        shortest_path_dict = dict()
        for child_node, level_parents in tree_dict.items():
            level_dict.setdefault(level_parents[0], []) \
                .append((child_node, level_parents[1]))

        for level in range(0, len(level_dict.keys())):
            for (child_node, parent_node_list) in level_dict[level]:
                if len(parent_node_list) > 0:
                    shortest_path_dict[child_node] = sum([shortest_path_dict[parent]
                                                          for parent in parent_node_list])
                else:
                    shortest_path_dict[child_node] = 1
        return shortest_path_dict

    def computeBetweenness(self):
        """
        compute betweenness of each edge pair
        :return: list of tuple(pair, float)
                => e.g. [(('0FVcoJko1kfZCrJRfssfIA', 'bbK1mL-AyYCHZncDQ_4RgA'), 189.0), ...
        """
        self.betweenness_result_dict = dict()
        # print('------------', self.vertexes[0])
        # bfs_tree = self._build_tree(root=self.vertexes[0])
        # temp_result_dict = self._traverse_tree(bfs_tree)
        # temp_result_list = sorted(temp_result_dict.items(), key=lambda kv: (-kv[1], kv[0][0]))
        # print(temp_result_list[:5])

        for node in self.vertexes:
            # 1.The algorithm begins by performing a breadth-first search
            # (BFS) of the graph, starting at the vertex X in all vertexes list
            # =>{'2GUjO7NU88cPXpoffYCU8w': (9, ['a48HhwcmjFLApZhiax41IA']),
            # '6YmRpoIuiq8I19Q8dHKTHw': (9, ['a48Hh
            bfs_tree = self._build_tree(root=node)

            # 2. Label each node by the number of shortest
            # paths that reach it from the root node
            # actually, this step has been done in the first step,
            # since the len of value[1] is exactly the number of shortest path
            # 3. Calculate for each edge e, the sum over all nodes
            # Y (of the fraction) of the shortest paths from the root
            # X to Y that go through edge e
            temp_result_dict = self._traverse_tree(bfs_tree)

            self.betweenness_result_dict = extend_dict(self.betweenness_result_dict,
                                                       temp_result_dict)

        # 4. Divide by 2 to get true betweenness
        self.betweenness_result_dict = \
            dict(map(lambda kv: (kv[0], float(kv[1] / 2)),
                     self.betweenness_result_dict.items()))

        self.betweenness_result_tuple_list = sorted(
            self.betweenness_result_dict.items(), key=lambda kv: (-kv[1], kv[0][0]))

        return self.betweenness_result_tuple_list

    def remove_edges(self):
        max_betweenness = self.betweenness_rlt[0][0]
        # print(self.betweenness_rlt[:3])
        for item in self.betweenness_rlt:
            if item[0] >= max_betweenness:
                edge = item[1]
                print('removing this edge: ', edge)
                print('edges storage detail: ',len(self.edges[edge[0]]), len(self.edges[edge[1]]))
                # self.edges[edge[0]].remove(edge[1])
                # self.edges[edge[1]].remove(edge[0])
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
                print('edges storage detail after remove: ', len(self.edges[edge[0]]), len(self.edges[edge[1]]))
            else: break

    def count_modularity(self):
        sum = 0
        for cluster in self.curr_communities:
            for pair in combinations(list(cluster), 2):
                pair = pair if pair[0] < pair[1] else (pair[1], pair[0])
                k0 = len(self.edges[pair[0]])
                k1 = len(self.edges[pair[1]])
                A_ij = 1 if pair[1] in self.adjacent_matrix[pair[0]] else 0
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
            print('\n')
            print(curr_modularity)
            print(self.betweenness_rlt[0])

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

if __name__ == '__main__':
    start = time.time()
    # define input variables
    # filter_threshold = "7"
    # input_csv_path = "../data/ub_sample_data.csv"
    # betweenness_file_path = "../out/task2_bet2.txt"
    # community_file_path = "../out/task2_com2.txt"

    filter_threshold = sys.argv[1]
    input_csv_path = sys.argv[2]
    betweenness_file_path = sys.argv[3]
    community_file_path = sys.argv[4]

    # conf = SparkConf().setMaster("local") \
    #     .setAppName("ay_hw_4_task2") \
    #     .set("spark.executor.memory", "4g") \
    #     .set("spark.driver.memory", "4g")
    sc = SparkContext()
    sparkSession = SparkSession(sc)
    sc.setLogLevel("WARN")

    # read the original json file and remove the header
    raw_data_rdd = sc.textFile(input_csv_path, 20)
    header = raw_data_rdd.first()
    uid_bidxes_dict = raw_data_rdd.filter(lambda line: line != header) \
        .map(lambda line: (line.split(',')[0], line.split(',')[1])) \
        .groupByKey().mapValues(lambda bids: sorted(list(bids))) \
        .collectAsMap()

    uid_pairs = list(itertools.combinations(list(uid_bidxes_dict.keys()), 2))

    edge_list = list()
    vertex_set = set()
    for pair in uid_pairs:
        if len(set(uid_bidxes_dict[pair[0]]).intersection(
                set(uid_bidxes_dict[pair[1]]))) >= int(filter_threshold):
            edge_list.append(tuple(pair))
            edge_list.append(tuple((pair[1], pair[0])))
            vertex_set.add(pair[0])
            vertex_set.add(pair[1])

    # => ['B7IvZ26ZUdL2jGbYsFVGxQ', 'jnn504CkjtfbYIwBquWmBw', 'sBqCpEUn0qYdpSF4Db
    vertexes = sc.parallelize(sorted(list(vertex_set))).collect()

    # => {'39FT2Ui8KUXwmUt6hnwy-g': ['0FVcoJko1kfZCrJRfssfIA', '1KQi8Ym
    edges = sc.parallelize(edge_list).groupByKey() \
        .mapValues(lambda uidxs: sorted(list(set(uidxs)))).collectAsMap()
    print('test edges: ', len(edges))
    print('test vertexes: ', len(vertexes))

    graph_frame = GraphFrame(vertexes, edges)
    betweenness_result = graph_frame.computeBetweenness()
    # export your finding
    export2File(betweenness_result, betweenness_file_path)

    # communities_result = graph_frame.extractCommunities()
    # # export your finding
    # export2File(communities_result, community_file_path)

    print("Duration: %d s." % (time.time() - start))