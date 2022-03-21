import sys
import time
from pyspark import SparkContext
import numpy as np
import xgboost
import json
import math


fold_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]
topN = 5


def read_csv(file_path, type):
    data = sc.textFile(file_path)
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
        return (user_id, business_id, 3.75, 0)  # predict by all users' average

    # count similarity
    ratings, similarities = [], []
    for i in range(len(user_items)):
        similar = count_pearson_similarity(business_id, user_items[i][0])
        if similar != 0:
            ratings.append(user_items[i][1])
            similarities.append(similar)

    # if no similar items, return all items that the particular user commented avg as prediction
    neighbor_size = len(similarities)
    if similarities == [] or len(similarities) < topN:
        avg = count_user_average_star(user_items)
        return (user_id, business_id, avg, neighbor_size)

    # chose top N similar items to predict
    similarity_rating = sorted(tuple(zip(similarities, ratings)), key=lambda x: x[0])[:topN]

    numerator, denominator = 0, 0
    for i in similarity_rating:
        if i[0] <= 0:
            break
        numerator += i[0] * i[1]
        denominator += abs(i[0])
    if numerator <= 25:
        avg = count_user_average_star(user_items)
        return (user_id, business_id, avg, neighbor_size)
    else:
        return (user_id, business_id, numerator / denominator, neighbor_size)


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


def generate_feature(rdd, type='train'):
    X = np.array(rdd.map(lambda r: find_feature_in_json(r)).collect())
    if type=='train':
        Y = np.array(rdd.map(lambda r: r[2]).collect())
        return X, Y
    return X


def find_feature_in_json(row):
    user_id = row[0]
    business_id = row[1]
    user_average_star, user_review_count, business_average_star, business_review_count = np.NAN, np.NAN, np.NAN, np.NAN
    if user_id in user_id_feature_map.keys():
        user_average_star = user_id_feature_map[user_id][0]
        user_review_count = user_id_feature_map[user_id][1]
    if business_id in business_id_feature_map.keys():
        business_average_star = business_id_feature_map[business_id][0]
        business_review_count = business_id_feature_map[business_id][1]
    return [user_average_star, user_review_count, business_average_star, business_review_count]


if __name__ == '__main__':
    start_time = time.time()

    sc = SparkContext.getOrCreate()
    train_rdd = read_csv(fold_path + 'yelp_train.csv', 'train')
    val_rdd = read_csv(test_file_path, 'val')
    # print('count test: ', val_rdd.count())
    user_set = set(train_rdd.map(lambda r: (r[0])).distinct().collect())
    business_set = set(train_rdd.map(lambda r: (r[1])).distinct().collect())
    # print(len(user_set), len(business_set))

    business_feature_rdd = sc.textFile(fold_path + "business.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], (float(x["stars"]), float(x["review_count"])))).filter(
        lambda x: x[0] in business_set)
    user_feature_rdd = sc.textFile(fold_path + "user.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["user_id"], (float(x["average_stars"]), float(x["review_count"])))).filter(
        lambda x: x[0] in user_set)
    # print(business_feature_rdd.first(), user_feature_rdd.first())
    business_id_feature_map = business_feature_rdd.collectAsMap()
    user_id_feature_map = user_feature_rdd.collectAsMap()

    X_train, Y_train = generate_feature(train_rdd)
    # print(np.shape(X_train))
    model = xgboost.XGBRegressor(n_estimators=30, random_state=233, max_depth=7)
    model.fit(X_train, Y_train)
    val_list = val_rdd.collect()
    X_pred = generate_feature(val_rdd, type='test')
    Y_pred = model.predict(X_pred)
    # print(train_rdd.first())

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
    user_item_dict = train_rdd.map(lambda r: (r[0], [r[1], float(r[2])])).groupByKey().mapValues(list).collectAsMap()
    # print('user_item_dict', list(user_item_dict.items())[0])

    # item similarity dic to avoid repeat and thus accelerate
    item_similarity_dic = {}

    CL_prediction = val_rdd.map(lambda r: item_based_collaborative_filter_with_neighbor_size(r[0], r[1])).collect()
    # print(CL_prediction[0])

    # max_neighbor = 0
    with open(output_file_path, 'w') as f:
        f.writelines('user_id, business_id, prediction' + '\n')
        for i in range(len(Y_pred)):
            model_based = Y_pred[i]
            item_based = CL_prediction[i][2]
            neighbor_size = CL_prediction[i][3]
            a = math.tanh(neighbor_size/300)
            final_score = a * item_based + (1 - a) * model_based
            f.writelines(str(val_list[i][0])+','+str(val_list[i][1])+','+str(final_score)+'\n')
            # max_neighbor = max(neighbor_size, max_neighbor)

    # print(max_neighbor)
    print('Duration: ', time.time() - start_time)
