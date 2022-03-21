import sys
import json
import datetime
review_path = sys.argv[1]
business_path = sys.argv[2]
output_a_path = sys.argv[3]
output_b_path = sys.argv[4]
# print(input_path, output_path)

from pyspark import SparkContext
sc = SparkContext('local')
review = sc.textFile(review_path).map(lambda r: json.loads(r))
business = sc.textFile(business_path).map(lambda r: json.loads(r))

# question a:
review_stars = review.map(lambda r: (r['business_id'], r['stars']))
business_city = business.map(lambda r: (r['business_id'], r['city']))
city_stars = business_city.fullOuterJoin(review_stars).map(lambda r: (r[1][0], r[1][1]))
city_ranks = city_stars.aggregateByKey((0,0), lambda u,v:(u[0]+float(v),u[1]+1), lambda u1, u2:(u1[0]+u2[0], u1[1]+u2[1])).\
    map(lambda r: (r[0], r[1][0]/r[1][1]))
ordered_ranks = city_ranks.takeOrdered(city_ranks.count(), key = lambda r: (-r[1], r[0]))
# print(ordered_ranks)
with open(output_a_path, 'w') as f1:
    f1.writelines('city,stars\n')
    for rank in ordered_ranks:
        f1.writelines(rank[0] + ',' + str(round(rank[1],1)) + '\n')

# question b:
ans_b = {}
# m1
start = datetime.datetime.now()
ordered_1 = city_ranks.takeOrdered(10, key = lambda r: (-r[1], r[0]))
print(ordered_1)
ans_b['m1'] = (datetime.datetime.now() - start).seconds
# m2
start = datetime.datetime.now()
rank_list = city_ranks.collect()
ordered_2 = sorted(rank_list, key = lambda i: (-i[1],i[0]))[10]
print(ordered_2)
ans_b['m2'] = (datetime.datetime.now() - start).seconds
ans_b['reason'] = 'm1 sort parallel int rdd'
with open(output_b_path, 'w') as f:
    json.dump(ans_b, f, indent=2)