from pyspark import SparkContext
import sys
import json
import time
import math
import pandas as pd
import numpy as np
import xgboost as xgb

folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]
sc = SparkContext.getOrCreate()

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

def get_distinct_idx(rawRDD, tag):
    if tag == 'user':
        return set(rawRDD.map(lambda x: x[0]).distinct().collect())
    elif tag == 'business':
        return set(rawRDD.map(lambda x: x[1]).distinct().collect())
    
def avg_rating_for_item(x):
    average = sum(i[1] for i in x[1]) / len(x[1])
    result = [(i[0], a[1]-avg) for a in x[1]]
    return x[0], dict([(i[0], i[1]) for i in x[1]]), dict(result)

def item_based_prediction(bid_uid, sim):
    # cold start
    if bid_uid[1] not in train_uid_basket:
        return re_uid_dict[bid_uid[1]] + ',' + re_bid_dict[bid_uid[0]] + ',' + str(3.75) + '\n'

    bsi_rating = train_uid_basket[bid_uid[1]]

    if bid_uid[0] not in train_uid_basket:
        ra = [x[1] for x in bsi_rating]
        avg_user = sum(ra)/len(ra)
        return re_uid_dict[bid_uid[1]] + ',' + re_bid_dict[bid_uid[0]] + ',' + str(avg_user) + '\n'

    sim_list = []
    rating_list = []
    to_rate = train_uid_basket[bid_uid[0]]
    
    for br in bsi_rating:
        ngbor = bid_basket_train[br[0]]
        corating_user = set(ngbor.keys()) & set(to_rate.keys())

        if len(corating_user)<3:
            continue
       
        pairs = tuple(sorted([br[0], bid_uid[0]]))
        
        if pairs in sim:
            s = sim[pairs]
            sim_list.append(s)
            rating_list.append(br[1])
        else:
            d_1 = pow(sum([pow(to_rate[u],2) for u in corating_user]),0.5)
            d_2 = pow(sum([pow(neighbors[u],2) for u in corating_user]),0.5)
            sim_denominator = d_1*d_2
            if sim_denominator == 0:
                sim[pairs] = 0
                continue
            s = sum([to_rate[u]*ngbor[u] for u in corating_user]) / sim_denominator
            if s<0:
                s += abs(s)*0.9
            else:
                s = s*1.1
            sim[pairs] = s
            sim_list.append(s)
            rating_list.append(br[1])
            
    if sim_list == []:
        o_ratings = list(train_bid_basket_ini[bid_uid[0]].values())
        other_avg_rating = sum(o_ratings)/len(o_ratings)
        return re_uid_dict[bid_uid[1]] + ',' + re_bid_dict[bid_uid[0]] + ',' + str(other_avg_rating) + '\n'
    elif len(sim_list)>=5:
        sim_rating = sorted(tuple(zip(sim_list,rating_list)), key = lambda x: -x[0])[:5]
        d = sum([abs(x[0]) for x in sim_rating])
        n = sum(x[0]*x[1] for x in sim_rating)
        if n <= 25:
            o_ratings = list(bid_basket_train_ini[bid_uid[0]].values())
            other_avg_rating = sum(other_ratings)/len(other_ratings)
            return re_uid_dict[bid_uid[1]] + ',' + re_bid_dict[bid_uid[0]] + ',' + str(other_avg_rating) + '\n'
        else:
            prediction = n / d
            return re_uid_dict[bid_uid[1]] + ',' + re_bid_dict[bid_uid[0]] + ',' + str(prediction) + '\n'
    else:
        o_ratings = list(train_bid_basket_ini[bid_uid[0]].values())
        other_avg_rating = sum(o_ratings)/len(o_ratings)
        return re_uid_dict[bid_uid[1]] + ',' + re_bid_dict[bid_uid[0]] + ',' + str(o_avg_r) + '\n'

def calculate_matrix(bid_uid,user_dataframe,bus_dataframe,tag):
    if tag=='train':
        a = list(user_dataframe.loc[bid_uid[1][0]]) + list(bus_dataframe.loc[bid_uid[0]])
        return a
    elif tag=='test':
        b = list(user_dataframe.loc[bid_uid[1][0]]) + list(bus_dataframe.loc[bid_uid[0]])
        return b

def get_RMSE(true_list, prediction_list):
    s = sum([pow(abs(true-pre),2) for true, pre in zip(true_list, prediction_list)])
    return pow((s/len(true_list)),0.5)

def hybrid_based_CF(folder_path,test_file_name,output_file_name):
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

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

    userRDD = sc.textFile(folder_path+'user.json', 30)
    data_user = userRDD.map(lambda x: json.loads(x)).map(lambda x: (x['user_id'],x['review_count'],x['average_stars'],x['useful'],x['fans'],x['funny'],x['cool']))
    data_user = data_user.filter(lambda x: x[0] in uid_dict).map(lambda x: (uid_dict[x[0]], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6]))).collect()
    
    user_dataframe = pd.DataFrame(data_user,columns=['user_id','review_count','average_stars', 'useful', 'fans','funny', 'cool'])
    user_dataframe.set_index('user_id',inplace=True)

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

    xgb_regression = xgb.XGBRegressor(verbosity=0, n_estimators=250, random_state=1, max_depth=7)
    xgb_regression.fit(x_train,y)

    x_test = np.array(test_rdd.map(lambda x: calculate_matrix(x, user_data, bus_data, 'test')).collect())
    y_prediction = xgb_regression.predict(x_test)

    train_uid_basket = trainRDD.map(lambda x: (x[1][0], (x[0], x[1][1])))
    train_uid_basket = train_uid_basket.groupByKey().map(lambda x: (x[0], list(x[1]))).collect()
    train_uid_basket = dict(zip([x[0] for x in train_uid_basket], [x[1] for x in train_uid_basket]))

    train_bid_basket_1 = trainRDD.groupByKey().map(lambda x: (x[0], list(x[1]))).map(de_avg).collect()

    train_bid_basket = dict(zip([x[0] for x in train_bid_basket_1,[x[2] for x in train_bid_basket_1]))
    train_bid_basket_ini = dict(zip([x[0] for x in train_bid_basket_1,[x[1] for x in train_bid_basket_1]))

    result = 'user_id, business_id, prediction\n'
    sim = {}

    result += testRDD.map(lambda x: item_based_prediction(x, sim)).reduce(lambda x, y: x+y)

    with open('tmp.csv', "w") as f:
        f.writelines(res)

    item_based_prediction = pd.read_csv('tmp.csv')
    value = pd.read_csv(folder_path+'yelp_val.csv')
    
    stars = list(value['stars'])
    xgb_stars = list(y_prediction)

    item_stars = list(item_based_prediction[' prediction'])
    weight_of_model = np.arange(0.6,1,0.005)

    r = 5
    RMSE_list = []
    for a in weight_of_model:
        w_i = 1-weight_of_model
        xgb_prediction = [i*w for i in xgb_stars]
        item_prediction = [j*w_i for j in item_stars]
        prediction = np.sum([xgb_prediction,pred_item], axis=0)
        RMSE = get_RMSE(stars, pred)
        RMSE_list.append(RMSE)
        if RMSE < r:
            r = RMSE
            final_prediction = prediction
            final_weight = a
   
    result = 'user_id, business_id, prediction\n'
    num = 0
    for j in testRDD.collect():
        result += re_uid_dict[j[1]] + ',' + re_bid_dict[j[0]] + ',' + str(final_prediction[num]) + '\n'
        num += 1

    with open(output_path, 'w') as f:
        f.writelines(result)

hybrid_based_CF(folder_path,test_file_name,output_file_name)
