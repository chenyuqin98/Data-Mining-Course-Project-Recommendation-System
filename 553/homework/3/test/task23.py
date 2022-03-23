import pyspark
from pyspark import SparkContext, SparkConf
import json
import sys
import math
from itertools import combinations
from operator import add
from datetime import datetime
import json
import random
import csv
from functools import reduce
import time
from collections import Counter, OrderedDict, defaultdict
import pandas as pd
import numpy as np
import xgboost as xgb

sc = SparkContext('local[*]', 'task').getOrCreate()
sc.setLogLevel('ERROR')

folderPath = sys.argv[1]
testFileName = sys.argv[2]
outputFileName = sys.argv[3]

start = time.time()

yelp_train = sc.textFile(folderPath + 'yelp_train.csv')
header = yelp_train.first()
yelp_train = yelp_train.filter(lambda data: data != header)
yelp_train = yelp_train.map(lambda data: data.split(','))
yelp_train = yelp_train.map(lambda data: [data[0], data[1], float(data[2])])

yelp_test = sc.textFile(testFileName)
header = yelp_test.first()
yelp_test = yelp_test.filter(lambda data: data != header)
yelp_test = yelp_test.map(lambda data: data.split(','))
yelp_test = yelp_test.map(lambda data: [data[0], data[1]])

userDictTrain = yelp_train.map(lambda data: data[0]).distinct().zipWithIndex().collectAsMap()
businessDictTrain = yelp_train.map(lambda data: data[1]).distinct().zipWithIndex().collectAsMap()

userDictTest = yelp_test.map(lambda data: data[0]).distinct().zipWithIndex().collectAsMap()
businessDictTest = yelp_test.map(lambda data: data[1]).distinct().zipWithIndex().collectAsMap()

userRDD = sc.textFile(folderPath + 'user.json')
userRDD = userRDD.map(lambda data: json.loads(data))
businessRDD = sc.textFile(folderPath + 'business.json')
businessRDD = businessRDD.map(lambda data: json.loads(data))

userRDD = userRDD.map(
    lambda data: (data['user_id'], data['average_stars'], data['review_count'], data['useful'], data['fans'])) \
    .filter(lambda data: data[0] in userDictTrain.keys() or data[0] in userDictTest.keys()) \
    .map(lambda data: (data[0], float(data[1]), float(data[2]), float(data[3]), float(data[4])))

businessRDD = businessRDD.map(lambda data: (data['business_id'], data['stars'], data['review_count'])) \
    .filter(lambda data: data[0] in businessDictTrain.keys() or data[0] in businessDictTest.keys()) \
    .map(lambda data: (data[0], float(data[1]), float(data[2])))

userCollection = userRDD.collect()
businessCollection = businessRDD.collect()

userDataFrame = pd.DataFrame(userCollection, columns=['userId', 'averageStar', 'reviewCount', 'useful', 'fans'])
businessDataFrame = pd.DataFrame(businessCollection, columns=['businessId', 'stars', 'reviewCount'])
userDataFrame.set_index('userId', inplace=True)
businessDataFrame.set_index('businessId', inplace=True)


def featureBuild(userId, businessId):
    userData = list(userDataFrame.loc[userId])
    businessData = list(businessDataFrame.loc[businessId])
    totalData = userData + businessData
    return totalData


xgbReg = xgb.XGBRegressor(verbosity=0, n_estimators=50, random_state=1, max_depth=7)

trainDataX = yelp_train.map(lambda data: featureBuild(data[0], data[1])).collect()
xgbTrainX = np.array(trainDataX)
trainDataY = yelp_train.map(lambda data: data[2]).collect()
xgbTrainY = np.array(trainDataY)

testDataX = yelp_test.map(lambda data: featureBuild(data[0], data[1])).collect()
xgbTestX = np.array(testDataX)

xgbReg.fit(xgbTrainX, xgbTrainY)

modelBasedResult = xgbReg.predict(xgbTestX)

coratedThreshold = 10
userDictTrain = yelp_train.map(lambda data: data[0]).distinct().zipWithIndex().collectAsMap()
businessDictTrain = yelp_train.map(lambda data: data[1]).distinct().zipWithIndex().collectAsMap()

userDictTest = yelp_test.map(lambda data: data[0]).distinct().zipWithIndex().collectAsMap()
businessDictTest = yelp_test.map(lambda data: data[1]).distinct().zipWithIndex().collectAsMap()


def averageRating(list):
    sum = 0
    for rating in list:
        sum += rating
    return float(sum) / len(list)


def convert(list):
    return {list[i][0]: list[i][1] for i in range(len(list))}


userRating = yelp_train.map(lambda data: (data[0], data[2])).groupByKey().mapValues(
    lambda data: averageRating(list(data))).collectAsMap()
businessRating = yelp_train.map(lambda data: (data[1], data[2])).groupByKey().mapValues(
    lambda data: averageRating(list(data))).collectAsMap()

startCount = len(userDictTrain)
for key in userDictTest.keys():
    if key not in userDictTrain.keys():
        userDictTrain[key] = startCount
        startCount += 1
userDict = userDictTrain
startCount = len(businessDictTrain)
for key in businessDictTest.keys():
    if key not in businessDictTrain.keys():
        businessDictTrain[key] = startCount
        startCount += 1
businessDict = businessDictTrain

ratingList = yelp_train.map(lambda data: data[2]).collect()
sum = 0
for i in ratingList:
    sum += float(i)
averageRatingInAll = sum / len(ratingList)

userBusinessRDD = yelp_train.map(lambda data: (data[0], (data[1], data[2]))).groupByKey().mapValues(
    lambda data: convert(list(data)))
businessUserRDD = yelp_train.map(lambda data: (data[1], (data[0], data[2]))).groupByKey().mapValues(
    lambda data: convert(list(data)))
userBusinessGroup = userBusinessRDD.collectAsMap()
businessUserGroup = businessUserRDD.collectAsMap()

unknownUser = set()
unknownBusiness = set()

for key in userDict.keys():
    if key not in userBusinessGroup:
        userBusinessGroup[key] = {}
        unknownUser.add(key)

for key in businessDict.keys():
    if key not in businessUserGroup:
        businessUserGroup[key] = {}
        unknownBusiness.add(key)

coratedThreshold = 20


def similarityCompute(businessId1, businessId2):
    userGroup1 = businessUserGroup[businessId1]
    userGroup2 = businessUserGroup[businessId2]

    userInCommon = list(set(userGroup1.keys()).intersection(set(userGroup2.keys())))
    if (len(userInCommon) <= coratedThreshold):
        return 0

    sumRating1 = 0
    for value in userGroup1.values():
        sumRating1 += float(value)
    averageRating1 = float(sumRating1) / len(userGroup1)

    sumRating2 = 0
    for value in userGroup2.values():
        sumRating2 += float(value)
    averageRating2 = float(sumRating2) / len(userGroup2)

    numerator = 0
    denominator1 = 0
    denominator2 = 0

    for user in userInCommon:
        numerator += (userGroup1[user] - averageRating1) * (userGroup2[user] - averageRating2)
        denominator1 += math.pow((userGroup1[user] - averageRating1), 2)
        denominator2 += math.pow((userGroup2[user] - averageRating2), 2)

    if denominator1 == 0 or denominator2 == 0:
        return 0

    denominator1 = math.sqrt(denominator1)
    denominator2 = math.sqrt(denominator2)

    similarity = float(numerator) / (denominator1 * denominator2)

    if (similarity < 0):
        return 0
    return similarity * 1.1


def hybridPredict(index, similarityList):
    similarityList = sorted(similarityList, key=lambda data: -data[0])
    numerator = 0
    denominator = 0

    selectItems = 0
    for i in range(len(similarityList)):
        if (similarityList[i][0] > 0.9):
            selectItems += 1
        else:
            break
    if (selectItems <= 5):
        return modelBasedResult[index]

    for i in range(selectItems):
        numerator += similarityList[i][0] * similarityList[i][1]
        denominator += abs(similarityList[i][0])
    if denominator == 0:
        return modelBasedResult[index]

    modelBasedRating = modelBasedResult[index]
    itemBasedRating = float(numerator) / denominator

    modelBasedParam = 10
    itemBasedParam = (selectItems - 5)
    alpha = float(itemBasedParam) / (modelBasedParam + itemBasedParam)

    return itemBasedRating * alpha + modelBasedRating * (1 - alpha)


hybridResult = yelp_test.zipWithIndex().map(lambda data: (data[0][0], data[0][1], data[1])) \
    .map(lambda data: (data[0], data[1], data[2], userBusinessGroup[data[0]])) \
    .map(lambda data: (data[0], data[1], data[2],
                       [(similarityCompute(data[1], businessId), rating) for businessId, rating in data[3].items()])) \
    .map(lambda data: (data[0], data[1], hybridPredict(data[2], data[3]))).collect()

with open(outputFileName, 'w+') as output:
    csvwriter = csv.writer(output, delimiter=',')
    csvwriter.writerow(['user_id', 'business_id', 'prediction'])
    for result in hybridResult:
        csvwriter.writerow([result[0], result[1], result[2]])
    output.close()

end = time.time()

print('the time cost is ' + str(end - start))