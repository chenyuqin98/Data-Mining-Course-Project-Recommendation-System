import sys
import time
from pyspark import SparkContext
import json
import numpy as np
import xgboost


fold_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]


def read_csv(file_path, type):
    data = sc.textFile(file_path)
    data_header = data.first()
    data = data.filter(lambda r: r != data_header)
    if type=='train':
        data = data.map(lambda r: (r.split(',')[0], r.split(',')[1], r.split(',')[2]))
    elif type=='val':
        data = data.map(lambda r: (r.split(',')[0], r.split(',')[1]))
    return data


def generate_feature(rdd, type='train'):
    X = np.array(rdd.map(lambda r: find_feature(r)).collect())
    if type=='train':
        Y = np.array(rdd.map(lambda r: r[2]).collect())
        return X, Y
    return X


def find_feature(row):
    user_id = row[0]
    business_id = row[1]
    user_average_star, user_review_count, business_average_star, business_review_count \
        = np.NAN, np.NAN, np.NAN, np.NAN
    user_useful, user_fans = np.NAN, np.NAN
    user_var, business_var = np.NAN, np.NAN
    if user_id in user_id_feature_map.keys():
        user_average_star = user_id_feature_map[user_id][0]
        user_review_count = user_id_feature_map[user_id][1]
        user_useful = user_id_feature_map[user_id][2]
        user_fans = user_id_feature_map[user_id][3]
    if business_id in business_id_feature_map.keys():
        business_average_star = business_id_feature_map[business_id][0]
        business_review_count = business_id_feature_map[business_id][1]
    if user_id in user_var_dict.keys():
        user_var = user_var_dict[user_id]
    if business_id in business_var_dict.keys():
        business_var = business_var_dict[business_id]
    # feature_list = [user_average_star, user_review_count, business_average_star, business_review_count,
    #                 user_var, business_var]
    # feature_list = [user_average_star, user_review_count, business_average_star, business_review_count]
    feature_list = [user_average_star, user_review_count, business_average_star, business_review_count,
                    user_useful, user_fans]
    return feature_list


def count_variance(row):
    item_list = row[1]
    sum, count = 0, 0
    for i in item_list:
        sum += i
        count += 1
    avg =  sum / count
    var = 0
    for i in item_list:
        var += (i - avg) ** 2
    var /= count
    return (row[0], var)


def count_RSME(k):
    with open("data/yelp_val.csv") as in_file:
        ans = in_file.readlines()[1:]
    RMSE = 0
    for i in range(len(Y_pred)):
        model_based = Y_pred[i]
        diff = float(model_based) - float(ans[i].split(",")[2])
        RMSE += diff ** 2
    RMSE = (RMSE / len(Y_pred)) ** (1 / 2)
    return RMSE


if __name__=='__main__':
    start_time = time.time()

    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    train_rdd = read_csv(fold_path+'yelp_train.csv', 'train')
    val_rdd = read_csv(test_file_path, 'val')

    # use 4 sets to filter json features.
    user_set = set(train_rdd.map(lambda r: (r[0])).distinct().collect())
    business_set = set(train_rdd.map(lambda r: (r[1])).distinct().collect())
    user_val_set = set(val_rdd.map(lambda r: (r[0])).distinct().collect())
    business_val_set = set(val_rdd.map(lambda r: (r[1])).distinct().collect())
    # print(len(user_set), len(business_set))

    # count variance features
    user_var = train_rdd.map(lambda r: (r[0], float(r[2]))).groupByKey().map(lambda r: count_variance(r))
    business_var = train_rdd.map(lambda r: (r[1], float(r[2]))).groupByKey().map(lambda r: count_variance(r))
    user_var_dict = user_var.collectAsMap()
    business_var_dict = business_var.collectAsMap()
    
    business_feature_rdd = sc.textFile(fold_path + "business.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], (float(x["stars"]), float(x["review_count"])))).filter(
        lambda x: x[0] in business_set or x[0] in business_val_set)
    user_feature_rdd = sc.textFile(fold_path + "user.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["user_id"], (float(x["average_stars"]), float(x["review_count"]),
        float(x['useful']), float(x['fans'])))).filter(
        lambda x: x[0] in user_set or x[0] in user_val_set)
    # print(business_feature_rdd.first(), user_feature_rdd.first())
    business_id_feature_map = business_feature_rdd.collectAsMap()
    user_id_feature_map = user_feature_rdd.collectAsMap()

    X_train, Y_train = generate_feature(train_rdd)
    # print(np.shape(X_train))

    alphas = [0.01, 0.1, 1, 3, 5, 10, 30, 100, 200]
    for n_est in range(10,300,10):
        for alpha in alphas:
            pass

    # use best parameters to build model
    model = xgboost.XGBRegressor(verbosity=0, n_estimators=50, random_state=1, max_depth=7)
    model.fit(X_train, Y_train)
    val_list = val_rdd.collect()
    X_pred = generate_feature(val_rdd, type='test')
    Y_pred = model.predict(X_pred)
    # print(len(X_pred), len(Y_pred), X_pred[0], Y_pred[0])

    with open(output_file_path, 'w') as f:
        f.writelines('user_id, business_id, prediction'+'\n')
        for i in range(len(Y_pred)):
            f.writelines(str(val_list[i][0])+','+ str(val_list[i][1])+','+str(Y_pred[i])+'\n')

    print('Duration: ', time.time() - start_time)