# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import sys
import time
from pyspark import SparkContext
import numpy as np
import xgboost
import json
import math
from collections import defaultdict
from datetime import datetime

fold_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]
topN = 5


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


def generate_feature(rdd, type='train'):
    X = np.array(rdd.map(lambda r: find_feature(r)).collect())
    if type == 'train':
        Y = np.array(rdd.map(lambda r: r[2]).collect())
        return X, Y
    return X


def find_feature(row):
    user_id = row[0]
    business_id = row[1]
    # business_average_star, business_review_count = np.NAN, np.NAN
    # user_var, business_var = np.NAN, np.NAN
    business_features = [np.NAN] * (len(business_numerical_features_name) + len(business_attribute_name))
    user_features = [np.NAN] * len(user_numerical_feature_name_list)
    if user_id in user_id_feature_map.keys():
        user_features = user_id_feature_map[user_id]
    if business_id in business_id_feature_map.keys():
        business_features = business_id_feature_map[business_id]
        # business_average_star = business_id_feature_map[business_id][0]
        # business_review_count = business_id_feature_map[business_id][1]
    # if user_id in user_var_dict.keys():
    #     user_var = user_var_dict[user_id]
    # if business_id in business_var_dict.keys():
    #     business_var = business_var_dict[business_id]
    feature_list = business_features + user_features
    return feature_list


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


def is_json(str):
    try:
        json.loads(str)
    except ValueError:
        return False
    return True


def find_attributes():
    for x in business_feature_list:
        attributes = x['attributes']
        # print(type(attributes))
        if attributes:
            for k, v in attributes.items():
                v = v.replace("'", '"').replace('False', 'false').replace('True', 'true')
                if is_json(v) and isinstance(json.loads(v), dict):
                    dic = json.loads(v)
                    # print(dic, type(dic))
                    for k, v in dic.items():
                        if isinstance(v, str):
                            business_attribute_dict[k].add(v)
                else:
                    business_attribute_dict[k].add(v)

        if x['city']:
            business_attribute_dict['city'].add(x['city'])
        if x['state']:
            business_attribute_dict['state'].add(x['state'])
        if x['postal_code']:
            business_attribute_dict['postal_code'].add(x['postal_code'])

        if x['categories']:
            categories_list = x['categories'].split(', ')
            for c in categories_list:
                category_words_num_dict[c] += 1


def encode_kv(k, v):
    value_list = sorted(list(business_attribute_dict[k]))
    try:
        en_k = business_attribute_name.index(k)
    except ValueError:
        en_k = -1
    try:
        en_v = value_list.index(v)
    except ValueError:
        en_v = -1
    return en_k, en_v


def encode_business_category(c):
    # return (category_frequency, category_index)
    category_frequency = -1
    if c in category_words_num_dict.keys():
        category_frequency = category_words_num_dict[c]
    try:
        category_index = category_words_list.index(c)
    except ValueError:
        category_index = -1
    return (category_frequency, category_index)


def encode_business_features(x):
    attributes_list = [np.NAN] * len(business_attribute_name)
    attributes = x['attributes']
    if attributes:
        for k, v in attributes.items():
            v = v.replace("'", '"').replace('False', 'false').replace('True', 'true')
            if is_json(v) and isinstance(json.loads(v), dict):
                dic = json.loads(v)
                for k, v in dic.items():
                    if isinstance(v, str):
                        en_k, en_v = encode_kv(k, v)
                        if en_k != -1:
                            attributes_list[en_k] = en_v
            else:
                en_k, en_v = encode_kv(k, v)
                if en_k != -1:
                    attributes_list[en_k] = en_v

    if x['city']:
        en_k, en_v = encode_kv('city', x['city'])
        attributes_list[en_k] = en_v
    if x['state']:
        en_k, en_v = encode_kv('state', x['state'])
        attributes_list[en_k] = en_v
    if x['postal_code']:
        en_k, en_v = encode_kv('postal_code', x['postal_code'])
        attributes_list[en_k] = en_v

    category_list = [np.NAN] * 3
    if x['categories']:
        categories_list = x['categories'].split(', ')
        # format: (category_frequency, category_index)
        categories_tuple_list = sorted(map(lambda x: encode_business_category(x), categories_list), reverse=True)
        for k, item in enumerate(categories_tuple_list):
            if k == 3: break
            frequency, numerical_categories = item
            category_list[k] = numerical_categories
    feature_list = [float(x[r]) if x[r] is not None else np.NAN for r in business_numerical_features_name]
    return feature_list + attributes_list + category_list
    # return feature_list + attributes_list


def encode_user_features(x):
    user_numerical_features = [float(x[r]) for r in user_numerical_feature_name_list]

    yelp_feature = np.NAN
    if x['yelping_since']:
        yelp_duration = datetime.now() - datetime.strptime(x['yelping_since'], '%Y-%m-%d')
        yelp_feature = yelp_duration.total_seconds() / (365 * 24 * 3600)  # year
        # print(yelp_duration, yelp_feature)

    friends_num = 0
    if x['friends'] and x['friends'] != 'None':
        friends_num = len(x['friends'].split(','))

    elite_num = 0
    if x['elite'] and x['elite'] != 'None':
        elite_num = len(x['elite'].split(','))

    return user_numerical_features + [yelp_feature, friends_num, elite_num]


def compute_metrics():
    with open(fold_path + "yelp_val.csv") as val_file:
        ans = val_file.readlines()[1:]
    res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, ">=4": 0}
    RMSE = 0
    for i in range(len(Y_pred)):
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
    print('Error Distribution:')
    print('>=0 and <1:', res["<1"])
    print('>=1 and <2:', res["1~2"])
    print('>=2 and <3:', res["2~3"])
    print('>=3 and <4:', res["3~4"])
    print('>=4:', res[">=4"], '\n')

    RMSE = (RMSE / len(Y_pred)) ** (1 / 2)
    print('RMSE:')
    print(RMSE, '\n')
    return RMSE


if __name__ == '__main__':
    start_time = time.time()
    description = 'The origin RMSE on valid data is 0.983970' + '\n' + \
                  '1. Use more user numerical features, RMSE decrease to 0.983621' + '\n' + \
                  '2. Use formula final_scores[i] = a * item_based + (1 - a) * model_based to combine, ' + '\n' + \
                  '   in which a = math.tanh(neighbor_size / k), train model and find the best k to combine model, ' + '\n' + \
                  '   RMSE decrease to 0.983612' + '\n' + \
                  '3. Most of business features are text format, encode them to numerical or bool (01), ' + '\n' + \
                  '   RMSE decrease to 0.980300' + '\n' + \
                  '4. Tune xgboost parameters (500 estimators, k = 25000), RMSE decrease to 0.977665' + '\n' + \
                  '5. Add business location features, RMSE 0.977643' + '\n' + \
                  '6. Encode business most frequent 3 categories features, RMSE 0.977747' + '\n' + \
                  '7. Encode user features: friends, elite, yelp_since, RMSE 0.977596 (local 0.977601)' + '\n' + \
                  '8. Update combine method' + '\n'
    print('Method Description:')
    print(description)

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

    # use this dict to encode text business attributes and location information (postcode...)
    business_attribute_dict = defaultdict(set)
    category_words_num_dict = defaultdict(int)
    business_feature_list = sc.textFile(fold_path + "business.json").map(lambda x: json.loads(x)).collect()
    # business_attribute_dict: {'BikeParking': {'true', 'false'}, ...}  key and all possible values
    find_attributes()
    business_attribute_name = sorted(list(business_attribute_dict.keys()))
    category_words_list = sorted([k for k, v in category_words_num_dict.items()])
    business_numerical_features_name = ["latitude", "longitude", "stars", "review_count", "is_open"]
    business_feature_rdd = sc.textFile(fold_path + "business.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], encode_business_features(x))).filter(
        lambda x: x[0] in business_set or x[0] in business_val_set)

    user_numerical_feature_name_list = ["review_count", "useful", "funny", "cool", "fans", "average_stars",
                                        "compliment_hot", "compliment_more", "compliment_profile",
                                        "compliment_cute", "compliment_list", "compliment_note", "compliment_plain",
                                        "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"]
    # eg: ["yelping_since":"2015-09-28","friends":"None","elite":"None",]
    user_text_feature_name_list = ["yelping_since", "friends", "elite"]
    user_feature_rdd = sc.textFile(fold_path + "user.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["user_id"], encode_user_features(x))) \
        .filter(lambda x: x[0] in user_set or x[0] in user_val_set)
    # print(business_feature_rdd.first(), user_feature_rdd.first())

    business_id_feature_map = business_feature_rdd.collectAsMap()
    user_id_feature_map = user_feature_rdd.collectAsMap()

    X_train, Y_train = generate_feature(train_rdd)
    # print(np.shape(X_train))
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 1, 'reg_lambda': 0}
    model = xgboost.XGBRegressor(**other_params)
    # model = xgboost.XGBRegressor(n_estimators=50, random_state=233, max_depth=7)
    model.fit(X_train, Y_train)
    val_list = val_rdd.collect()
    X_pred = generate_feature(val_rdd, type='test')
    Y_pred = model.predict(X_pred)
    # print(train_rdd.first())

    # combine two algorithms
    if 1 == 1:
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

        CF_train_pred = train_rdd.map(lambda r: item_based_collaborative_filter_with_neighbor_size(r[0], r[1])).collect()
        CF_prediction = val_rdd.map(lambda r: item_based_collaborative_filter_with_neighbor_size(r[0], r[1])).collect()

        # count final scores
        final_scores = [0] * len(Y_pred)
        for i in range(len(Y_pred)):
            model_based = Y_pred[i]
            item_based = CF_prediction[i][2]
            neighbor_size = CF_prediction[i][3]
            a = math.tanh(neighbor_size / 25000)
            final_scores[i] = a * item_based + (1 - a) * model_based

    RMSE = compute_metrics()

    with open(output_file_path, 'w+') as f:
        f.writelines('user_id, business_id, prediction' + '\n')
        for i in range(len(Y_pred)):
            f.writelines(str(val_list[i][0]) + ',' + str(val_list[i][1]) + ',' + str(final_scores[i]) + '\n')

    with open('2_models_results.csv', 'w+') as f:
        # f.writelines('CF_prediction, Y_pred, a' + '\n')
        for i in range(len(Y_pred)):
            neighbor_size = CF_prediction[i][3]
            a = math.tanh(neighbor_size / 500)
            f.writelines(str(CF_prediction[i][2])+','+str(Y_pred[i])+','+str(a)+'\n')

    # print(max_neighbor)
    print('Execution Time:')
    print(time.time() - start_time)
