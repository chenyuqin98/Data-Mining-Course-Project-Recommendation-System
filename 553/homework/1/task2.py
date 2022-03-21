import sys
import json
import datetime
input_path = sys.argv[1]
output_path = sys.argv[2]
n_partition = sys.argv[3]
# print(input_path, output_path)

from pyspark import SparkContext
sc = SparkContext('local')
ans = {}

ans['default'] = {}
start = datetime.datetime.now()
review = sc.textFile(input_path).map(lambda r: json.loads(r)).map(lambda r: (r['business_id'], 1))
review = review.repartition(int(n_partition))
ans['default']['n_partition'] = review.getNumPartitions()
ans['default']['n_items'] = review.glom().map(len).collect()
top10_business = review.reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda r: (-r[1], r[0]))
ans['default']['exe_time'] = (datetime.datetime.now() - start).seconds

ans['customized'] = {}
start = datetime.datetime.now()
review = sc.textFile(input_path).map(lambda r: json.loads(r)).map(lambda r: (r['business_id'], 1))
review = review.partitionBy(int(n_partition), lambda k: ord(k[:1]))
ans['customized']['n_partition'] = review.getNumPartitions()
ans['customized']['n_items'] = review.glom().map(len).collect()
top10_business2 = review.reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda r: (-r[1], r[0]))
ans['customized']['exe_time'] = (datetime.datetime.now() - start).seconds

# print(ans)

with open(output_path, 'w') as f:
    json.dump(ans, f, indent=2)
