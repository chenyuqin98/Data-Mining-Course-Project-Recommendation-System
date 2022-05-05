# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import numpy as np
from pyspark import SparkContext
import json
from datetime import datetime
from collections import defaultdict

def encode_user_features(x):
    user_numerical_features = [float(x[r]) for r in user_numerical_feature_name_list]

    yelp_feature = np.NAN
    if x['yelping_since']:
        yelp_duration = datetime.now() - datetime.strptime(x['yelping_since'], '%Y-%m-%d')
        yelp_feature = yelp_duration.total_seconds() / (365*24*3600) # year
        # print(yelp_duration, yelp_feature)

    friends_num = 0
    if x['friends'] and x['friends'] != 'None':
        friends_num = len(x['friends'].split(','))

    elite_num = 0
    if x['elite'] and x['elite'] != 'None':
        elite_num = len(x['elite'].split(','))

    return user_numerical_features + [yelp_feature, friends_num, elite_num]

if __name__ == '__main__':
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    user_numerical_feature_name_list = ["review_count", "useful", "funny", "cool", "fans", "average_stars",
                                        "compliment_hot", "compliment_more", "compliment_profile",
                                        "compliment_cute", "compliment_list", "compliment_note", "compliment_plain",
                                        "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"]
    # eg: ["yelping_since":"2015-09-28","friends":"None","elite":"None",]
    user_text_feature_name_list = ["yelping_since", "friends", "elite"]
    user_feature_rdd = sc.textFile("data/user.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["user_id"], encode_user_features(x)))
    print(user_feature_rdd.collect()[0])