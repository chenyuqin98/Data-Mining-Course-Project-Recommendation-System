# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import sys
import time
from pyspark import SparkContext
from tqdm import tqdm

fold_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]
# topN = 5


def read_csv(file_path, type):
    data = sc.textFile(file_path)
    data.count()  # pyspark version problem
    data_header = data.first()
    data = data.filter(lambda r: r != data_header)
    if type == 'train':
        data = data.map(lambda r: (r.split(',')[0], r.split(',')[1], r.split(',')[2]))
    elif type == 'val':
        data = data.map(lambda r: (r.split(',')[0], r.split(',')[1]))
    return data


def count_user_average_star(user_items):
    # user_items: [(business_id, star)...]
    sum, total = 0, 0
    for items in user_items:
        sum += items[1]
        total += 1
    return sum / total


def item_based_collaborative_filter_with_neighbor_size(user_id, business_id):
    user_items = user_item_dict[user_id]  # [(business_id, star)...] all items that the particular user commented

    # for new items, predict by user average
    if business_id not in item_user_dict.keys():
        if len(user_items) > 0:  # if this has comment on other items
            avg = count_user_average_star(user_items)
            return (user_id, business_id, avg, 0)
        return (user_id, business_id, 3.75, 0, [-1] * topN)  # predict by all users' average

    # count similarity
    ratings, similarities = [], []
    for i in range(len(user_items)):
        similar = count_pearson_similarity(business_id, user_items[i][0])
        if similar > 0.5:
            ratings.append(user_items[i][1])
            similarities.append(similar)

    # if no similar items, return all items that the particular user commented avg as prediction
    neighbor_size = len(similarities)
    if similarities == [] or len(similarities) < topN:
        avg = count_user_average_star(user_items)
        similarity_feature = sorted(similarities + [-1] * (len(similarities) - topN), reverse=True)
        return (user_id, business_id, avg, neighbor_size, similarity_feature)

    # chose top N similar items to predict
    similarity_rating = sorted(tuple(zip(similarities, ratings)), key=lambda x: x[0], reverse=True)[:topN]
    similarity_feature = sorted(similarities)[:topN]

    numerator, denominator = 0, 0
    for i in similarity_rating:
        if i[0] <= 0:
            break
        numerator += i[0] * i[1]
        denominator += abs(i[0])
    if numerator <= 25:
        avg = count_user_average_star(user_items)
        return (user_id, business_id, avg, neighbor_size, similarity_feature)
    else:
        return (user_id, business_id, numerator / denominator, neighbor_size, similarity_feature)


def count_pearson_similarity(item1, item2):
    key_value = tuple(sorted([item1, item2]))
    if key_value in item_similarity_dic.keys():
        return item_similarity_dic[key_value]

    # using all-rating average seems easier
    # item1 is the item we want to predict
    numerator, item1_len, item2_len = 0, 0, 0
    co_users = item_user_dict[item1] & item_user_dict[item2]

    if len(co_users) < 3:
        # print('no co-user between item1 and item2')
        return 0

    # co-ratings:
    # star1_sum, star2_sum, star1_cnt, star2_cnt = 0, 0, 0, 0
    # for u in co_users:
    #     if (u, item1) in train_star_dict.keys() and (u, item2) in train_star_dict:
    #         star1_sum += train_star_dict[(u, item1)]
    #         star2_sum += train_star_dict[(u, item2)]
    #         star1_cnt += 1
    #         star2_cnt += 1
    # star1_avg = star1_sum / star1_cnt
    # star2_avg = star2_sum / star2_cnt

    for u in co_users:
        # if (u, item1) in train_star_dict.keys() and (u, item2) in train_star_dict:
        star1 = train_star_dict[(u, item1)] - item_avg_dict[item1]  # all-ratings:
        star2 = train_star_dict[(u, item2)] - item_avg_dict[item2]
        # star1 = train_star_dict[(u, item1)] - star1_avg  # co-ratings:
        # star2 = train_star_dict[(u,item2)] - star2_avg
        numerator += star1 * star2
        item1_len += pow(star1, 2)
        item2_len += pow(star2, 2)
        # else:
        #     print(11111111111111)
    if numerator != 0:
        item_similarity_dic[key_value] = numerator / (pow(item1_len, 0.5) * pow(item2_len, 0.5))
        return numerator / (pow(item1_len, 0.5) * pow(item2_len, 0.5))

    # print('divided by 0: ', co_users)
    # for u in co_users:
    #     star1 = train_star_dict[(u,item1)] - item_avg_dict[item1] # all-ratings:
    #     star2 = train_star_dict[(u,item2)] - item_avg_dict[item2]
    #     print(star1, star2, '\n')
    #     numerator += star1*star2
    #     item1_len += pow(star1, 2)
    #     item2_len += pow(star2, 2)
    return 0


def count_item_avg(row):
    item_list = row[1]
    sum, count = 0, 0
    for i in item_list:
        sum += i
        count += 1
    return (row[0], sum / count)


def count_variance(row):
    item_list = row[1]
    sum, count = 0, 0
    for i in item_list:
        sum += i
        count += 1
    avg = sum / count
    var = 0
    for i in item_list:
        var += (i - avg) ** 2
    var /= count
    return (row[0], var)


def compute_metrics():
    with open(fold_path + "yelp_val.csv") as val_file:
        ans = val_file.readlines()[1:]
    res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, ">=4": 0}
    RMSE = 0
    for i in range(len(CF_prediction)):
        diff = float(final_scores[i]) - float(ans[i].split(",")[2])
        RMSE += diff ** 2
        if abs(diff) < 1:
            res["<1"] = res["<1"] + 1
        elif abs(diff) < 2:
            res["1~2"] = res["1~2"] + 1
        elif abs(diff) < 3:
            res["2~3"] = res["2~3"] + 1
        elif abs(diff) < 4:
            res["3~4"] = res["3~4"] + 1
        else:
            res[">=4"] = res[">=4"] + 1
    # print('Error Distribution:')
    # print('>=0 and <1:', res["<1"])
    # print('>=1 and <2:', res["1~2"])
    # print('>=2 and <3:', res["2~3"])
    # print('>=3 and <4:', res["3~4"])
    # print('>=4:', res[">=4"], '\n')

    RMSE = (RMSE / len(CF_prediction)) ** (1 / 2)
    # print('RMSE:')
    # print(RMSE, '\n')
    return RMSE


if __name__ == '__main__':
    start_time = time.time()
    topN = 5

    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    train_rdd = read_csv(fold_path + 'yelp_train.csv', 'train')
    val_rdd = read_csv(test_file_path, 'val')

    # use 4 sets to filter json features.
    user_set = set(train_rdd.map(lambda r: (r[0])).distinct().collect())
    business_set = set(train_rdd.map(lambda r: (r[1])).distinct().collect())
    user_val_set = set(val_rdd.map(lambda r: (r[0])).distinct().collect())
    business_val_set = set(val_rdd.map(lambda r: (r[1])).distinct().collect())

    # count variance features
    user_var = train_rdd.map(lambda r: (r[0], float(r[2]))).groupByKey().map(lambda r: count_variance(r))
    business_var = train_rdd.map(lambda r: (r[1], float(r[2]))).groupByKey().map(lambda r: count_variance(r))
    user_var_dict = user_var.collectAsMap()
    business_var_dict = business_var.collectAsMap()

    # using all-rating average to normalize
    # eg: {'3MntE_HWbNNoyiLGxywjYA': 3.4}
    train_item_avg = train_rdd.map(lambda r: (r[1], float(r[2]))).groupByKey().map(lambda r: count_item_avg(r))
    item_avg_dict = train_item_avg.collectAsMap()
    # print(train_item_avg.first())
    # print('item_avg_dict', list(item_avg_dict.items())[0])

    # collect item user dict to seek co-item
    item_user_dict = train_rdd.map(lambda r: (r[1], r[0])).groupByKey().mapValues(set).collectAsMap()
    # print('item_user_dict', list(item_user_dict.items())[0])

    # transfer the triple to dict for quickly find: (user_id,business_id,stars) -> {(user_id, business_id) : stars}
    train_star_dict = train_rdd.map(lambda r: ((r[0], r[1]), float(r[2]))).collectAsMap()
    # print('train_star_dict', list(train_star_dict.items())[0])

    # to find items both stared by one user. what we get is user co-items: {user_id: [(business_id, star)...]}
    user_item_dict = train_rdd.map(lambda r: (r[0], [r[1], float(r[2])])).groupByKey().mapValues(
        list).collectAsMap()
    # print('user_item_dict', list(user_item_dict.items())[0])

    # item similarity dic to avoid repeat and thus accelerate
    item_similarity_dic = {}

    best_N = 3
    best_RMSE = 100
    RMSE_list = []
    for topN in tqdm(range(3, 10)):
        CF_prediction = val_rdd.map(lambda r: item_based_collaborative_filter_with_neighbor_size(r[0], r[1])).collect()

        # count final scores
        final_scores = [0] * len(CF_prediction)
        for i in range(len(CF_prediction)):
            item_based = CF_prediction[i][2]
            final_scores[i] = item_based

        RMSE = compute_metrics()
        RMSE_list.append(RMSE)
        if RMSE < best_RMSE:
            best_RMSE = RMSE
            best_N = topN
    print('RMSE:', best_RMSE, 'best_N:', best_N)

    # print(max_neighbor)
    print('Execution Time:')
    print(time.time() - start_time)
