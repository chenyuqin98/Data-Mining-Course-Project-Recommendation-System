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
bands_num = 50
rows_num = 1

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
    fileRDD.count()
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
    rawRDD = fileRDD.map(lambda x: (dict_business_brc.value[x[0]], dict_business_brc.value[x[1]]))
    join_RDD = rawRDD.groupByKey().mapValues(set)
    join_RDD.count()
    join_dict = join_RDD.collectAsMap()
    join_dict_brc = sc.broadcast(join_dict)    

    # signature_rdd
    business_sig_RDD = join_RDD.mapValues(lambda x:min_hashing(hash_para_a,hash_para_b,user_num,x)).mapValues(lambda x: split_sig_to_bands(x))
    business_sig = business_sig_RDD.collectAsMap()
    business_sig_brc = sc.broadcast(business_sig)
    
    business_pairsRDD = sc.parallelize([i for i in range(business_num)]).flatMap(lambda x: [(x, j) for j in range(x + 1, business_num)])    
    

    def similar_pairs(pair1,pair2):
        sig1 = business_sig_brc.value[pair1]
        sig2 = business_sig_brc.value[pair2]
        for j in range(bands_num):
            if sig1[j] == sig2[j]:
                return True
        return False

    def get_jaccard(pair1,pair2):
        set1 = join_dict_brc.value[pair1]
        set2 = join_dict_brc.value[pair2]
        intersection = set1 & set2
        union = set1 | set2
        similarity = float(len(intersection)/len(union))
        return similarity

    candi_RDD = business_pairsRDD.filter(lambda x: similar_pairs(x[0], x[1])) 
    resultRDD = candi_RDD.map(lambda x: (x[0], x[1], get_jaccard(x[0], x[1]))).filter(lambda x: x[2] >= similarity_threshold).map(lambda y: (idx_business_brc.value[y[0]], idx_business_brc.value[y[1]], y[2]))
    results = resultRDD.collect()
    print("#similar pairs:", len(results))

    out = {}
    for pairs in results:
        out["business_id_1"] = pairs[0]
        out["business_id_2"] = pairs[1]
        out["similarity"] = pairs[2]

    with open(output_file_path, 'w+') as f:
        json.dump(out,f)
        f.close()
        
start_time = time.time()
lsh(input_file_path, output_file_path)
end_time = time.time()
print("Runtime: {0:.2f}".format(end_time - start_time))
    
