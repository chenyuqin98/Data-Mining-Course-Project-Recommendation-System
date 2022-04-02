from pyspark import SparkContext
import sys
import json
import time
import random
from itertools import combinations

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]   

similarity_threshold = 0.5
hash_num = 50
bands_num = 25
rows_num = 2

hash_para_a = []
hash_para_b = []

for i in range(hash_num):
    hash_para_a.append(random.randint(1000, 9999))
    hash_para_b.append(random.randint(1000, 9999))

def min_hashing(hash_para_a,hash_para_b,user_num,user_set):
    idx = [user_set]
    for i in range(hash_num - 1):
        l = map(lambda y:(hash_para_a[i]*y+hash_para_b[i]) % user_num, user_set)
        idx.append(set(l))

    min_idx = [min(j) for j in idx]
    return min_idx

def split_sig_to_bands(signature):
    res_list = []
    for j in range(bands_num):
        res_list.append(signature[j*rows_num:(j+1)*rows_num])
    return res_list

def lsh(input_file_path, output_file_path):
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    # format into RDD
    fileRDD = sc.textFile(input_file_path).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    head = fileRDD.first()
    fileRDD = fileRDD.filter(lambda x: x!=head)
    preuserRDD = fileRDD.map(lambda x: x[0])
    userRDD = preuserRDD.distinct().zipWithIndex()
    dict_of_user = userRDD.collectAsMap()
    user_num = len(dict_of_user)
    dict_user_brc = sc.broadcast(dict_of_user)

    prebusRDD = fileRDD.map(lambda x: x[1])
    businessRDD = prebusRDD.distinct().zipWithIndex()
    dict_of_business = businessRDD.collectAsMap()
    business_num = len(dict_of_business)
    dict_business_brc = sc.broadcast(dict_of_business)
    idx_business_dict = businessRDD.map(lambda x: (x[1], x[0])).collectAsMap()
    idx_business_brc = sc.broadcast(idx_business_dict)

    # combine two rdd
    rawRDD = fileRDD.map(lambda x: (dict_business_brc.value[x[1]], dict_user_brc.value[x[0]]))
    join_RDD = rawRDD.groupByKey().mapValues(set)
    join_dict = join_RDD.collectAsMap()
    for k,v in join_dict.items():
        print('join_dict', k, v)
        break
    join_dict_brc = sc.broadcast(join_dict)

    # signature_rdd
    business_sig_RDD = join_RDD.mapValues(lambda x:min_hashing(hash_para_a,hash_para_b,user_num,x)).mapValues(lambda x: split_sig_to_bands(x))
    business_sig = business_sig_RDD.collectAsMap()
    for k,v in business_sig.items():
        print('business_sig', k,v)
        break

    candidate_RDD = business_sig_RDD.flatMap(lambda row: [(tuple(sorted(c)), row[0]) for c in row[1]]).groupByKey()\
        .map(lambda row: list(row[1])).filter(lambda row: len(row) > 1)
    print('candidate_RDD fist: ', candidate_RDD.first())

    business_sig_brc = sc.broadcast(business_sig)

    # business_pairsRDD = sc.parallelize([i for i in range(business_num)]).flatMap(lambda x: [(x, j) for j in range(x + 1, business_num)])


    def similar_pairs(list):
        rlt = [p for p in combinations(list, 2)]
        # print('test rlt -------', rlt, list)
        for r in rlt:
            if r[0] != r[1]:
                yield tuple(sorted(r))


    # def similar_pairs(pair1,pair2):
    #     sig1 = business_sig_brc.value[pair1] # throw error
    #     sig2 = business_sig_brc.value[pair2]
    #     for j in range(bands_num):
    #         if sig1[j] == sig2[j]:
    #             return True
    #     return False

    def get_jaccard(pair1,pair2):
        set1 = join_dict_brc.value[pair1]
        set2 = join_dict_brc.value[pair2]
        intersection = set1 & set2
        union = set1 | set2
        similarity = float(len(intersection)/len(union))
        return similarity

    # print(business_pairsRDD.first(), business_pairsRDD.count())
    # business_pairsRDD (0, 1)
    # candi_RDD = business_pairsRDD.filter(lambda x: similar_pairs(x[0], x[1]))

    candi_RDD = candidate_RDD.flatMap(lambda r: similar_pairs(r))
    print('candi_RDD', candi_RDD.first(), candi_RDD.count())

    resultRDD = candi_RDD.map(lambda x: (x[0], x[1], get_jaccard(x[0], x[1])))\
        .filter(lambda x: x[2] >= similarity_threshold).distinct().\
        map(lambda y: (idx_business_brc.value[y[0]], idx_business_brc.value[y[1]], y[2]))\
        .map(lambda y: (sorted([y[0], y[1]]), y[2])).sortBy(lambda r:r[0]).\
        map(lambda y: (y[0][0], y[0][1], y[1]))
    results = resultRDD.collect()
    print("#similar pairs:", len(results))

    with open(output_file_path, 'w') as f:
        f.writelines('business_id_1, business_id_2, similarity' + '\n')
        for r in results:
            f.writelines(r[0] + ',' + r[1] + ',' + str(r[2]) + '\n')
        
start_time = time.time()
lsh(input_file_path, output_file_path)
end_time = time.time()
print("Runtime: {0:.2f}".format(end_time - start_time))
    
