import sys
import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
from itertools import combinations
from graphframes import GraphFrame
import time

filter_threshold = sys.argv[1]
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]
# os.environ["PYSPARK_SUBMIT_ARGS"] = ("pyspark-shell --driver-memory 4G"
#                                      "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")


def filter_pairs():
    edges = list()
    vertexes = set()
    for pair in candidate_pairs:
        business1 = user_business_dict[pair[0]]
        business2 = user_business_dict[pair[1]]
        if len(business1 & business2) >= int(filter_threshold):
            # print(type(business1))
            vertexes.add(pair[0])
            vertexes.add(pair[1])
            edges.append(tuple(pair))
            edges.append(tuple((pair[1], pair[0])))
    return edges, list(vertexes)


if __name__ == "__main__":
    start_time = time.time()
    sc = SparkContext.getOrCreate()
    sparkSession = SparkSession(sc)
    sc.setLogLevel("WARN")

    # read the original json file and remove the header
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
    edges_df = sc.parallelize(edges).toDF(["src", "dst"])
    vertexes_df = sc.parallelize(vertexes).map(lambda r: (r,)).toDF(["id"])
    # print(edges[0], len(edges))
    # print(vertexes[0], len(vertexes))

    graph = GraphFrame(vertexes_df, edges_df)
    community = graph.labelPropagation(maxIter=5)
    rlt = community.rdd.map(lambda r: (r[1], r[0])).groupByKey().map(lambda r: sorted(list(r[1])))\
        .sortBy(lambda r: (len(r), r)).collect()
    # dummy graph
    # rlt = user_business.filter(lambda r: len(list(r[1]))>1).map(lambda r: (len(r[1]), r[0]))\
    #     .groupByKey().map(lambda r: sorted(list(r[1]))).sortBy(lambda r: (len(r), r)).collect()
    # print('first 5: ------------', rlt[:5], len(rlt))
    # print('mid', rlt[int(len(rlt)/2)])
    # print('all rlt: ------------')
    # for r in rlt:
    #     print(r)

    with open(community_output_file_path, 'w') as f:
        for r in rlt:
            write_r = str(r).strip('[').strip(']') + '\n'
            f.writelines(write_r)
            # f.writelines(str(r)[1:-1]+'\n')

    print('Duration: ', time.time() - start_time)