from pyspark import SparkContext
import sys

print(sys.version, sys.executable)
sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')
data = sc.textFile("C://Users//yuqin//Desktop//2022spring_coursework//553//homework//3//data//yelp_train.csv")
print(data.count(), data.first())
# data = data.filter(lambda r: r != data_header)
# print(data.collect())
print('finish')