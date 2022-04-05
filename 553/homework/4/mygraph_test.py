import collections
import queue
from tqdm import tqdm

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

    def dfs_count_betweenness(self, node, depth = 0, curr_edge=None):
        node_val = 1
        # print('debug dfs_count_betweenness:', node, len(self.bfs_tree[node]))
        for next_node in self.bfs_tree[node]:
            node_val += self.dfs_count_betweenness(next_node, depth+1, (node, next_node))

        contribute = 0 # root
        if curr_edge is not None:
            contribute = node_val * self.shortest_path_num[curr_edge[0]] / self.shortest_path_num[node]
            if curr_edge[0] > curr_edge[1]:
                curr_edge = (curr_edge[1], curr_edge[0])
            self.betweenness[curr_edge] += contribute
        return contribute

    def count_betweennesses(self):
        # self.bfs_count_shortest_path_num('e')
        # self.dfs_count_betweenness('e')
        for i in range(len(self.vertexes)):
            node = self.vertexes[i]
            self.bfs_count_shortest_path_num(node)
            self.dfs_count_betweenness(node)
        for k in self.betweenness.keys():
            self.betweenness[k] /= 2

if __name__ == "__main__":
    vertexes = ['a','b','c','d','e','f','g']
    edges = [('a','b'),('a','c'),
             ('b','a'),('b','c'),('b','d'),
             ('c','a'),('c','b'),
             ('d','b'),('d','e'),('d','g'),('d','f'),
             ('e','d'),('e','f'),
             ('f', 'd'),('f', 'e'),('f', 'g'),
             ('g','d'),('g','f')]
    my_graph = MyGraph(edges, vertexes)
    print('test parent_num: ', my_graph.shortest_path_num, len(my_graph.shortest_path_num))
    print('test betweenness: ', my_graph.betweenness, len(my_graph.betweenness))
    betweenness_rlt = []
    for k in my_graph.betweenness.keys():
        if k[0] < k[1]:
            betweenness_rlt.append((float(my_graph.betweenness[k]), k))
    betweenness_rlt.sort(key=lambda r: (-r[0], r[1]))
    print('test betweenness_rlt: ', betweenness_rlt)