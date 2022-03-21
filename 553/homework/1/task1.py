import sys
import json
input_path = sys.argv[1]
output_path = sys.argv[2]
# print(input_path, output_path)

from pyspark import SparkContext
sc = SparkContext('local')
review = sc.textFile(input_path).map(lambda r: json.loads(r))
ans = {}

ans['n_review'] = review.count()
ans['n_review_2018'] = review.filter(lambda r:r['date'].split('-')[0]=='2018').count()
ans['n_user'] = review.map(lambda r:r['user_id']).distinct().count()
ans['top10_user'] = review.map(lambda r: (r['user_id'], 1)).reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda r: (-r[1], r[0]))
ans['n_business'] = review.map(lambda r:r['business_id']).distinct().count()
ans['top10_business'] = review.map(lambda r: (r['business_id'], 1)).reduceByKey(lambda a,b: a+b).takeOrdered(10, key=lambda r: (-r[1], r[0]))
print(ans)

with open(output_path, 'w') as f:
    json.dump(ans, f, indent=2)
