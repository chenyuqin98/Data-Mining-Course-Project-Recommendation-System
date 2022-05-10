# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
from pyspark import SparkContext
import json
from collections import defaultdict


def parse_business_feature(x):
    feature_list = [float(x[r]) if x[r] is not None else 0 for r in business_digital_features_name_list]
    return (x["business_id"], feature_list)

def is_json(str):
    try:
        json.loads(str)
    except ValueError:
        return False
    return True

def to_json(s):
    # print(s)
    s = s.replace("'", '"').replace('"{', '{').replace('}"', '}')
    # print(s)
    # return s
    return json.loads(s)

def find_attributes():
    total_c, count_c = 0, 0
    for x in business_feature_list:
        attributes = x['attributes']
        # print(type(attributes))
        if attributes:
            for k, v in attributes.items():
                v = v.replace("'", '"').replace('False', 'false').replace('True', 'true')
                if is_json(v) and isinstance(json.loads(v), dict):
                    dic = json.loads(v)
                    # print(dic, type(dic))
                    for k, v in dic.items():
                        if isinstance(v, str):
                            distinct_attribute[k].add(v)
                else:
                    distinct_attribute[k].add(v)
        postal_code = x['postal_code']
        if postal_code:
            distinct_attribute['postal_code'].add(postal_code)
        if x['categories']:
            categories_list = x['categories'].split(', ')
            # total_c += len(categories_list)
            # count_c += 1
            for c in categories_list:
                category_words_num[c] += 1
    # print(total_c/count_c)


if __name__ == '__main__':
    distinct_attribute = defaultdict(set)
    category_words_num = defaultdict(int)
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    # business_feature = sc.textFile("data/business.json").map(lambda x: json.loads(x)).\
    #     map(lambda x: parse_business_feature(x))
    business_feature = sc.textFile("../data/business.json").map(lambda x: json.loads(x))
    business_feature_list = business_feature.collect()
    business_digital_features_name_list = ["latitude", "longitude", "stars", "review_count", "is_open"]
    if 1 == 1:
        a = {"business_id": "Apn5Q_b6Nz61Tq4XzPdf9A", "name": "Minhas Micro Brewery", "neighborhood": "",
         "address": "1314 44 Avenue NE", "city": "Calgary", "state": "AB", "postal_code": "T2E 6L6",
         "attributes": {"BikeParking": "False", "BusinessAcceptsCreditCards": "True",
                        "BusinessParking": "{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}",
                        "GoodForKids": "True", "HasTV": "True", "NoiseLevel": "average", "OutdoorSeating": "False",
                        "RestaurantsAttire": "casual", "RestaurantsDelivery": "False", "RestaurantsGoodForGroups": "True",
                        "RestaurantsPriceRange2": "2", "RestaurantsReservations": "True", "RestaurantsTakeOut": "True"},
         "categories": "Tours, Breweries, Pizza, Restaurants, Food, Hotels & Travel",
         "hours": {"Monday": "8:30-17:0", "Tuesday": "11:0-21:0", "Wednesday": "11:0-21:0", "Thursday": "11:0-21:0",
                   "Friday": "11:0-21:0", "Saturday": "11:0-21:0"}}
        b = {"business_id": "AjEbIBw6ZFfln7ePHha9PA", "name": "CK'S BBQ & Catering", "neighborhood": "", "address": "",
         "city": "Henderson", "state": "NV", "postal_code": "89002", "latitude": 35.9607337, "longitude": -114.939821,
         "stars": 4.5, "review_count": 3, "is_open": 0,
         "attributes": {"Alcohol": "none", "BikeParking": "False", "BusinessAcceptsCreditCards": "True",
                        "BusinessParking": "{'garage': False, 'street': True, 'validated': False, 'lot': True, 'valet': False}",
                        "Caters": "True", "DogsAllowed": "True", "DriveThru": "False", "GoodForKids": "True",
                        "GoodForMeal": "{'dessert': False, 'latenight': False, 'lunch': False, 'dinner': False, 'breakfast': False, 'brunch': False}",
                        "HasTV": "False", "OutdoorSeating": "True", "RestaurantsAttire": "casual",
                        "RestaurantsDelivery": "False", "RestaurantsGoodForGroups": "True", "RestaurantsPriceRange2": "2",
                        "RestaurantsReservations": "False", "RestaurantsTableService": "False",
                        "RestaurantsTakeOut": "True", "WheelchairAccessible": "True", "WiFi": "no"},
         "categories": "Chicken Wings, Burgers, Caterers, Street Vendors, Barbeque, Food Trucks, Food, Restaurants, Event Planning & Services",
         "hours": {"Friday": "17:0-23:0", "Saturday": "17:0-23:0", "Sunday": "17:0-23:0"}}
    # print(business_feature.collect()[0])

    find_attributes()
    print(distinct_attribute, len(distinct_attribute), len(distinct_attribute['post_code']))

    category_words_num_list = sorted([(v,k) for k, v in category_words_num.items()], reverse=True)
    print(len(category_words_num_list), category_words_num_list[100])
