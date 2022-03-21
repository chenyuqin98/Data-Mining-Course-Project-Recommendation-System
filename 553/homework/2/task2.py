import sys
import time
from pyspark import SparkContext
import collections
from itertools import combinations
from functools import reduce

filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]
BUCKET_NUM = 100


def preprocess(row):
    row_list = row.split(',')
    TRANSACTION_DT = row_list[0].strip('"')
    CUSTOMER_ID = str(int(row_list[1].strip('"')))
    # PRODUCT_ID = int(row_list[5].strip('"'))
    PRODUCT_ID = str(int(row_list[5].strip('"')))
    DATE_CUSTOMER_ID = TRANSACTION_DT[:-4] + TRANSACTION_DT[-2:] + '-' + CUSTOMER_ID
    yield (DATE_CUSTOMER_ID, PRODUCT_ID)


def hash_function(pair):
    total = 0
    for p in pair:
        total += int(p)
    return total % BUCKET_NUM


def gen_next_candidates(candidates_list, curr_size):
    next_candidates = []
    n = len(candidates_list)
    for i in range(n):
        for j in range(i + 1, n):
            pair_1, pair2 = candidates_list[i], candidates_list[j]
            # print('pair_1, pair2', pair_1, pair2)
            combination = tuple(sorted(list(set(pair_1).union(set(pair2)))))
            # print('comb', combination)
            if len(combination) == curr_size + 1:
                temp_candidates = []
                for c in combinations(combination, curr_size):
                    # print('c in comb', c)
                    temp_candidates.append(c)
                # print('set: ', temp_candidates, set(temp_candidates))
                if set(temp_candidates).issubset(set(candidates_list)):
                    next_candidates.append(combination)
            else:
                break
    return next_candidates


def find_candidates(baskets, support, total_num):
    # use PCY to find candidates
    baskets_list = list(baskets) # convert map object to list
    # init local threshold, bitmap, counter
    local_support_threshold = support * len(baskets_list) / total_num
    # print(len(baskets_list), local_support_threshold, total_num)
    bitmap = [False for _ in range(BUCKET_NUM)]
    bit_counter = [0 for _ in range(BUCKET_NUM)]
    counter = collections.defaultdict(int)
    # PCY pass 1:
    for b in baskets_list:
        for item in b:
            counter[item] += 1
        for pair in combinations(b, 2):
            hash_val = hash_function(pair)
            bit_counter[hash_val] += 1
    # print('counter:', counter, bit_counter)
    single_frequent_set = set()
    for i in counter.items():
        if i[1] > local_support_threshold:
            single_frequent_set.add(i[0])
    single_frequent = sorted(list(single_frequent_set))
    for i in range(BUCKET_NUM):
        if bit_counter[i] > local_support_threshold:
            bitmap[i] = True
    # print('bitmap', bitmap)
    # print('single_frequent', single_frequent)

    # only consider single items
    filtered_baskets = []
    for b in baskets_list:
        filtered_baskets.append(sorted(list(set(b).intersection(single_frequent_set))))
    all_candidate_dict = collections.defaultdict(list)
    # print('single_frequent', single_frequent)
    # all_candidate_dict[1] = single_frequent
    all_candidate_dict[1] = [tuple([item]) for item in single_frequent]
    # print('all_candidate_dict', all_candidate_dict)

    # PCY pass 2:
    curr_candidates = single_frequent
    # print(curr_candidates)
    curr_pair_len = 2 # current pairs contain 2 items
    # repeat pass 2, until curr_candidates is none
    while len(curr_candidates) > 0:
        counter2 = collections.defaultdict(int)
        for b in filtered_baskets:
            if len(b) >= curr_pair_len:
                if curr_pair_len==2:
                    for pair in combinations(b, 2):
                        hash_val = hash_function(pair)
                        if bitmap[hash_val]: counter2[pair] += 1
                else:
                    for candidate_item in curr_candidates:
                        if set(candidate_item).issubset(set(b)):
                            counter2[candidate_item] += 1
        # filter curr_candidates
        # filtered_pairs = dict(filter(lambda r: r[1] >= local_support_threshold, counter2.items()))
        filtered_candidates = dict(filter(lambda r: r[1] >= local_support_threshold, counter2.items()))
        # generate new candidate list
        curr_candidates = gen_next_candidates(sorted(list(filtered_candidates.keys())), curr_pair_len)
        if len(filtered_candidates) == 0:
            break
        all_candidate_dict[str(curr_pair_len)] = list(filtered_candidates)
        curr_pair_len += 1

    yield reduce(lambda val1, val2: val1 + val2, all_candidate_dict.values())


def check_candidates(data, candidates):
    for c in candidates:
        if set(c).issubset(set(data)):
            yield [(c, 1)]


def rdd2string(rdd_list):
    string = ""
    curr_len = 1
    for c in rdd_list:
        # print(c)
        if len(c) == 1:
            string += str("(" + str(c)[1:-2] + "),")
        elif len(c) > curr_len:
            string = string[:-1]
            string += "\n\n"
            curr_len = len(c)
            string += (str(c) + ",")
        else:
            string += (str(c) + ",")
    return string[:-1]


if __name__=='__main__':
    start_time = time.time()

    sc = SparkContext.getOrCreate()
    partition_num = 2
    data_rdd = sc.textFile(input_file_path, partition_num)
    header = data_rdd.first()
    data_rdd = data_rdd.filter(lambda r : r != header)
    # print(data_rdd.collect()[0])
    shopping_record = data_rdd.map(lambda r: preprocess(r)).flatMap(lambda r:r)
    # print(shopping_record.collect()[0])

    baskets = shopping_record.groupByKey().map(lambda r: sorted(list(set(r[1])))).filter(lambda r: len(r)>filter_threshold)
    data_num = baskets.count()
    # print(baskets.collect(), data_num)

    candidates = baskets.mapPartitions(lambda r: find_candidates(baskets=r, support=support, total_num=data_num))
    candidates = candidates.flatMap(lambda pairs: pairs).distinct().sortBy(lambda pairs: (len(pairs), pairs)).collect()
    print(candidates, len(candidates))
    frequent = baskets.flatMap(lambda r:check_candidates(r, candidates)).flatMap(lambda r:r).reduceByKey(lambda x, y: x+y)\
        .filter(lambda r:r[1]>=support).map(lambda r: r[0]).sortBy(lambda pairs: (len(pairs), pairs)).collect()
    print(frequent, len(frequent))

    with open(output_file_path, 'w') as f:
        f.write('Candidates:\n')
        f.write(rdd2string(candidates)+'\n\n')
        f.write('Frequent Itemsets:\n')
        f.write(rdd2string(frequent))

    print('Duration: ', time.time() - start_time)