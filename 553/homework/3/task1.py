import sys
import time
from pyspark import SparkContext
import random
import itertools


input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

b = 25
r = 2
# b = 30
# r = 1
function_num = b*r
threshold = 0.5
hash_para_a = random.sample(range(1, sys.maxsize - 1), function_num)
hash_para_b = random.sample(range(0, sys.maxsize - 1), function_num)


def min_hash(user_set, user_num):
    min_hash_li = [sys.maxsize - 1] * function_num
    for i in range(function_num):
        for x in user_set:
            min_hash_li[i] = min(min_hash_li[i], ((hash_para_a[i] * x + hash_para_b[i]) % 233333333333) % user_num)
    return min_hash_li


def LSH_partition(hash_val_li):
    portions = []
    for i in range(b):
        portions.append(hash_val_li[i*r:(i+1)*r])
    return portions


def compute_jaccard(business1, business2):
    s1 = business_user_dic[business1]
    s2 = business_user_dic[business2]
    jaccard = float(float(len(s1 & s2)) / float(len(s1 | s2)))
    return jaccard


if __name__=='__main__':
    start_time = time.time()

    # read data to spark
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    all_rdd = sc.textFile(input_file_path)
    header = all_rdd.first()
    business_user = all_rdd.filter(lambda r: r != header).map(lambda r: (r.split(',')[1], r.split(',')[0])).sortByKey()

    # create self-defined user id dict
    user_id = business_user.map(lambda x: x[1]).distinct().zipWithIndex()
    user_dict = user_id.collectAsMap()
    # print('test user num', len(user_dict))

    # # replace user with self defined id
    business_user = business_user.map(lambda x: (x[0], user_dict[x[1]]))
    business_user = business_user.groupByKey().mapValues(set)
    # print('test business_user', business_user.first())
    business_user_dic = business_user.collectAsMap()

    # min hash
    business_signature = business_user.mapValues(lambda x: min_hash(x, len(user_dict)))
    print(business_signature.first())

    # LSH
    business_signature = business_signature.mapValues(lambda x:LSH_partition(x))
    print(business_signature.first())

    candidate_pairs = business_signature.flatMap(lambda r: [(tuple(sorted(chunk)), r[0]) for chunk in r[1]]).groupByKey()\
        .map(lambda r: list(r[1])).filter(lambda r: len(r)>1).flatMap(lambda r: [p for p in itertools.combinations(r, 2)])\
        .map(lambda r: tuple(sorted(list(r)))).distinct().filter(lambda r: r[0]!=r[1])
    # print('first candidate pair:', candidate_pairs.collect()[0])
    print('b:', b, 'r:', r, ', count candidates:', candidate_pairs.count(), ', time:', time.time() - start_time)

    resultRDD = candidate_pairs.map(lambda x: (x[0], x[1], compute_jaccard(x[0], x[1]))).filter(lambda x: x[2] >= threshold)
    results = resultRDD.sortBy(lambda r:r).collect()
    print('result number:', resultRDD.count())

    with open(output_file_path, 'w') as f:
        f.writelines('business_id_1, business_id_2, similarity'+'\n')
        for r in results:
            f.writelines(r[0]+','+r[1]+','+str(r[2])+'\n')

    print('Duration: ', time.time() - start_time)