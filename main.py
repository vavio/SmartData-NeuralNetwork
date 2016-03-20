
from pymongo import MongoClient
from time import mktime

import time
import datetime


def save_to_mongodb():
    line_number = 0
    client = MongoClient()
    db = client.smart_grid

    db_cache = []
    print 'PARSING DATA...'
    for l in open('dataset.txt'):
        line_number += 1

        if line_number == 1:
            continue

        if line_number % 10000 == 0:
            print '\tIDX {0}'.format(line_number)
            db.data.insert_many(db_cache)
            db_cache = []

        data = l.split('\t')[:3]
        db_cache.append({
            'customer_id': int(data[0]),
            'datetime': datetime.datetime.fromtimestamp(mktime(time.strptime(data[1], "%d/%m/%Y %H:%M"))),
            'supply_usage': float(data[2])})

    if len(db_cache) != 0:
        db.data.insert_many(db_cache)

    print 'PARSING DATA FINISHED'


def load_data():
    # TODO: Read the data from mongodb
    # TODO: Generate and return [test, train, validation] set
    return []


def train_nn(train_set):
    # TODO: Train NN with input [time of the day, w_day, y_day] and output [supply_usage]
    # TODO: Also consider for input the usage in the same day as the previous week
    pass


start_time = time.time()
save_to_mongodb()
datasets = load_data()
train_nn(datasets[0])
print time.time() - start_time
