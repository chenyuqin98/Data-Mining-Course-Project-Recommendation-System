from pyspark import SparkContext
import sys
import json
import time
import math
import xgboost as xgb
import pandas as pd
import numpy as np

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]


def get_distinct_idx(rawRDD, tag):
    if tag == 'user':
        return set(rawRDD.map(lambda x: x[0]).distinct().collect())
    elif tag == 'business':
        return set(rawRDD.map(lambda x: x[1]).distinct().collect())

def get_RMSE(real_list,prediction_list):
    he = sum([pow(abs(real-prediction),2) for real, prediction in zip(real_list, prediction_list)])
    return pow((he/len(real_list)),0.5)

def calculate_matrix(bid_uid,user_dataframe,bus_dataframe,tag):
    if tag=='train':
        a = list(user_dataframe.loc[bid_uid[1][0]]) + list(bus_dataframe.loc[bid_uid[0]])
        return a
    elif tag=='test':
        b = list(user_dataframe.loc[bid_uid[1][0]]) + list(bus_dataframe.loc[bid_uid[0]])
        return b
    
def model_based_CF(folder_path,test_file_name,output_file_name):
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    def load_data(train_file_name,tag):
        rawRDD = sc.textFile(train_file_name,30)
        head = rawRDD.first()
        rawRDD = rawRDD.filter(lambda x: x!=head)
        if tag =='train':
        # (user, business, rating)
            rawRDD = rawRDD.map(lambda x: (x.split(",")[0], x.split(",")[1], x.split(",")[2]))
            return rawRDD
        elif tag == 'test':
            rawRDD = rawRDD.map(lambda x: (x.split(",")[0], x.split(",")[1]))
            return rawRDD

    trainRDD = load_data(folder_path+'yelp_train.csv', 'train')
    testRDD = load_data(test_file_name, 'test')

    train_user_set = get_distinct_idx(trainRDD, 'user')
    test_user_set = get_distinct_idx(testRDD, 'user')
    full_user  = train_user_set & test_user_set
    uid_dict = dict(zip(full_user, range(len(full_user))))
    re_user_id_dict = dict(zip(range(len(full_user)), full_user))

    train_business_set = get_distinct_idx(train_rdd, 'business')
    test_business_set = get_distinct_idx(test_rdd, 'business')
    full_business = train_business_set & test_business_set
    bid_dict = dict(zip(full_business, range(len(full_business))))
    re_business_id_dict = dict(zip(range(len(full_business)), full_business))
    
    bid_len = len(full_business)
    uid_len = len(full_user)

    # (bid, (uid,rating))
    trainRDD = trainRDD.map(lambda x: (bid_dict[x[1]], (uid_dict[x[0]], float(x[2])))).cache()
    # (bid, uid)
    testRDD = testRDD.map(lambda x: (bid_dict[x[1]], uid_dict[x[0]])).cache()

    # create features
    userRDD = sc.textFile(folder_path+'user.json', 30)
    data_user = userRDD.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'],x['review_count'],x['average_stars'],x['useful'],x['fans'],x['funny'],x['cool']))
    data_user = data_user.filter(lambda x: x[0] in uid_dict).map(lambda x: (uid_dict[x[0]], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6]))).collect()
    
    user_dataframe = pd.DataFrame(data_user,columns=['user_id','review_count','average_stars', 'useful', 'fans','funny', 'cool'])
    user_dataframe.set_index('user_id',inplace=True)

    # business.json
    businessRDD = sc.textFile(folder_path+'business.json', 30)
    data_business = businessRDD.map(lambda x: json.loads(x)).map(lambda x: (x['business_id'],x['stars'],x['review_count']))
    data_business = data_busines.filter(lambda x: x[0] in bid_dict)
    data_business = data_business.map(lambda x: (business_id_dict[x[0]], float(x[1]), float(x[2]))).collect()
    bus_dataframe = pd.DataFrame(data_business, columns=['business_id','stars','review_count'])
    bus_dataframe.set_index('business_id',inplace=True)

    bus_idx = set(bus_dataframe.index)
    user_idx = set(user_dataframe.index)

    x_train = np.array(trainRDD.map(lambda x: calculate_matrix(x, user_dataframe, bus_dataframe, 'train')).collect())
    y = np.array(trainRDD.map(lambda x: x[1][1]).collect())

    # model training
    yelp_val = pd.read_csv(folder_path+'yelp_val.csv')
    stars = list(yelp_val['stars'])

    x_test = np.array(testRDD.map(lambda x: calculate_matrix(x, user_dataframe, bus_dataframe, 'test')).collect())
    n_estimator = range(250,290,5)
    max_depth = range(7,8)
    min_r = 5

    for i in n_estimator:
        for j in max_depth:
            xgb_regression = xgb.XGBRegressor(verbosity=0, n_estimators=e, random_state=1, max_depth=d)
            xgb_regression.fit(x_train,y)
            y_prediction = xgb_regression.predict(x_test)
            RMSE = get_RMSE(stars,list(y_prediction))
            if RMSE < min_r:
                min_r = RMSE
                result_est = i
                result_depth = j
                final_prediction = y_prediction

    # output_format
    def output(y_prediction,output_file_name):
        result = 'user_id, business_id, prediction\n'
        n = 0
        for i in testRDD.collect():
            result += reversed_user_id_dict[i[1]] + ',' + reversed_business_id_dict[i[0]] + ',' + str(y_prediction[n]) + '\n'
            n += 1
        with open(output_file_name, 'w') as f:
            f.writelines(result)

    output(y_prediction,output_file_name)

model_based_CF(folder_path,test_file_name,output_file_name)
                
