from pyspark import SparkContext
import sys
import json
import time
import math
import csv

train_file_name = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

corating_threshold = 10
topn = 2

avg_bsi_rating = 3.7512
avg_user_rating = 3.7512


def get_distinct_idx(tag,rawRDD):
    if tag == 'user':
        groupby_RDD = rawRDD.map(lambda x: (x.split(",")[0], 1)).reduceByKey(lambda a, b: a)
        res = groupby_RDD.map(lambda x: (1,[x[0]])).reduceByKey(lambda x,y: x+y)
        get_result = res.collect()[0][1]
    elif tag == 'business':
        groupby_RDD = rawRDD.map(lambda x: (x.split(",")[1], 1)).reduceByKey(lambda a, b: a)
        res = groupby_RDD.map(lambda x: (1,[x[0]])).reduceByKey(lambda x,y: x+y)
        get_result = res.collect()[0][1]
    return get_result

def merge_dict(x, y):
    x.update(y)
    return x

def get_similarity(business_dict,idx1, idx2):
    bid1_basket = business_dict[idx1]
    bid2_basket = business_dict[idx2]

    if len(bid1_basket)==0 or len(bid2_basket)==0:
        return 0
    else:
        avg_bid1 = sum(bid1_basket.values())/len(bid1_basket)
        avg_bid2 = sum(bid2_basket.values())/len(bid2_basket)
        
        corating_user = list(set(bid1_basket.keys()) & set(bid2_basket.keys()))
        if len(corating_user) <= 1:
            return 0
        if len(corating_user) < corating_threshold:
            return 0.05

        num = 0
        div_bid1 = 0
        div_bid2 = 0
        
        for u in corating_user:
            num += (bid1_basket[u] - avg_bid1) * (bid2_basket[u] - avg_bid2)
            div_bid1 += pow(bid1_basket[u] - avg_bid1, 2)
            div_bid2 += pow(bid2_basket[u] - avg_bid2, 2)

        if div_bid1 == 0 or div_bid2 == 0:
            return 0
        
        similarity = num / (pow(div_bid1,0.5)*pow(div_bid2,0.5))
        if similarity < 0:
            similarity = similarity + abs(similarity) * 0.9
        else:
            similarity = similarity * 1.1
        return similarity

x = [(0.1,2),(0.2,3)]
x_1 = []
if len(x_1)==0:
    print('1')

def rating_prediction(user_idx, bsi_idx, l):
    # l is a list of (similarity,rating)
    num = 0
    d = 0
    
    if len(l)==0:
        return avg_user_rating
    else:
        sorted_l = sorted(l, key = lambda x: -x[0])
        top_n = topn
        if sorted_l[0][0]>0.8 or sorted_l[1][0]>0.8:
            top_n = len(sorted_l)
        for i in range(topn):
            num += sorted_l[i][0]*sorted_l[i][1]
            d += abs(sorted_l[i][0])
        if d == 0:
            return avg_user_rating
        else:
            return num/d

def export_file(prediction, output_file_name):
    with open(output_file_name,'w+') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(['user_id','business_id','prediction'])
        for rows in prediction:
            w.writerow([rows[0],rows[1],rows[2]])
        f.close()

def item_based_CF(train_file_name,test_file_name,output_file_name):
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    def load_data(train_file_name):
        rawRDD = sc.textFile(train_file_name,30)
        head = rawRDD.first()
        rawRDD = rawRDD.filter(lambda x: x!=head)
        return rawRDD

    # read_data
    train_data = load_data(train_file_name)
    test_data = load_data(test_file_name)

    # get distinct
    train_uid = get_distinct_idx('user', train_data)
    test_uid = get_distinct_idx('user', test_data)
    uid_all = list(set(train_uid + test_uid))

    train_bid = get_distinct_idx('business', train_data)
    test_bid = get_distinct_idx('business',test_data)
    bid_all = list(set(train_bid + test_bid))

    # generate index
    uid_dict = {}
    for idx, uid in enumerate(uid_all):
        uid_dict[uid] = idx
    bid_dict = {}
    for idx, bid in enumerate(bid_all):
        bid_dict[bid] = idx

    # create user basket
    user_dict = train_data.map(lambda x: (uid_dict[x.split(',')[0]], {bid_dict[x.split(',')[1]]: float(x.split(',')[2])})).reduceByKey(lambda x,y: merge_dict(x,y))
    user_dict = user_dict.map(lambda x: (1, {x[0]:x[1]})).reduceByKey(lambda x,y: merge_dict(x,y))
    user_dict = user_dict.map(lambda x: x[1]).collect()[0]

    for u in uid_dict.values():
        if u not in user_dict.keys():
            user_dict[u] = {}

    # create business dict
    business_dict = train_data.map(lambda x: (bid_dict[x.split(',')[1]], {uid_dict[x.split(',')[0]]: float(x.split(',')[2])})).reduceByKey(lambda x,y: merge_dict(x,y))
    business_dict = business_dict.map(lambda x: (1, {x[0]:x[1]})).reduceByKey(lambda x,y: merge_dict(x,y))
    business_dict = business_dict.map(lambda x: x[1]).collect()[0]

    for bid in bid_dict.values():
        if bid not in business_dict.keys():
            business_dict[bid] = {}

    idx2user = {v:k for k,v in uid_dict.items()}
    idx2bsi = {v:k for k,v in bid_dict.items()}

    prediction = test_data.map(lambda x: (uid_dict[x.split(',')[0]], bid_dict[x.split(',')[1]])).map(lambda x: (x[0], x[1], user_dict[x[0]]))
    prediction = prediction.map(lambda x: (x[0], x[1], [(get_similarity(business_dict,x[1], bid), rating) for bid, rating in x[2].items()]))
    prediction = prediction.map(lambda x: (idx2user[x[0]], idx2bsi[x[1]], rating_prediction(x[0],x[1],x[2]))).collect()

    # output the results
    export_file(prediction, output_file_name)

item_based_CF(train_file_name,test_file_name,output_file_name)
