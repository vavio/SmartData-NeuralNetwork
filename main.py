# dataset: https://data.gov.au/dataset/sample-household-electricity-time-of-use-data/resource/ed7aaa03-6282-4254-9dcb-0e80bc6dc90d?inner_span=True
#  Organisations Department of Industry, ... Sample household electricity ... Smart Grid Smart Cities Data ...

from pymongo import MongoClient
from datetime import datetime, timedelta, datetime
from sklearn import preprocessing
from sknn.mlp import Regressor, Layer

import time
import numpy as np


def convert_time(hours, minutes):
    return hours * 60 + minutes


def preprocess_data(input_file='dataset.txt'):
    line_number = 0
    customer_map = {}
    week_subtract = timedelta(days=7)

    print 'PARSING DATA...'
    data = []
    for l in open(input_file):
        line_number += 1

        if line_number == 1:
            continue

        if line_number % 50000 == 0:
            print '\tIDX {0}'.format(line_number)

        raw_data = l.split('\t')[:3]

        customer_id = raw_data[0]
        isodate = datetime.strptime(raw_data[1], "%d/%m/%Y %H:%M")
        supply_usage = float(raw_data[2])

        if customer_id not in customer_map:
            customer_map[customer_id] = {}

        # if isodate in customer_map[customer_id]:
        #     print line_number, customer_id, isodate, customer_map[customer_id][isodate]
        customer_map[customer_id][isodate] = supply_usage

        previous_week = isodate - week_subtract
        previous_week_usage = customer_map[customer_id].get(previous_week, 0)
        # TODO: Possibly add usage in previous hours

        data.append([convert_time(isodate.hour, isodate.minute), isodate.weekday(), isodate.timetuple().tm_yday,
                     previous_week_usage, supply_usage])

    print 'NORMALIZING DATA...'
    data = np.array(data)
    N = data.shape[0]
    X = preprocessing.scale(data[:, :-1]).reshape(N, data.shape[1])
    Y = np.array(data[:, -1])

    client = MongoClient()
    db = client.smart_grid

    db_cache = []
    print 'SAVING RAW DATA TO MONGODB...'
    for cust_id in customer_map:
        print '\tCUSTOMER_ID {0}'.format(cust_id)
        for cust_date in customer_map[customer_id]:
            db_cache.append({
                'customer_id': cust_id,
                'datetime': cust_date,
                'supply_usage': customer_map[customer_id][cust_date]
            })

        print len(db_cache)
        db.user_info.insert_many(db_cache)
        db_cache = []

    print 'SAVING NORMALIZED DATA TO MONGODB'
    db_cache = []
    for i in xrange(N):
        if (i + 1) % 50000 == 0:
            print 'IDX {0}'.format(i + 1)
            db.user_data.insert_many(db_cache)
            db_cache = []

        current_data = X[i, :]
        db_cache.append({
            'time_of_day': current_data[0],
            'week_day': current_data[1],
            'year_day': current_data[2],
            'previous_usage': current_data[3],
            'current_usage': Y[i]
        })

    if len(db_cache) != 0:
        db.user_data.insert_many(db_cache)


def get_vector(data):
    return [
        1.0,  # bias
        data['time_of_day'],
        data['week_day'],
        data['year_day'],
        data['previous_usage'],
        data['current_usage']
    ]


def load_data():
    client = MongoClient()
    db = client.smart_grid

    trainX = []
    trainY = []
    testX = []
    testY = []
    validationX = []
    validationY = []

    print 'READING DATA...'
    idx = 0
    for data in db.user_data.find({}):

        if idx % 100000 == 0:
            print '\tIDX {0}'.format(idx)

        idx += 1

        rnd = np.random.rand()
        vec = np.array(get_vector(data))
        X = vec[:-1]
        y = vec[-1]

        if rnd < 0.5:
            trainX.append(X)
            trainY.append(y)
        elif rnd < 0.75:
            testX.append(X)
            testY.append(y)
        else:
            validationX.append(X)
            validationY.append(y)

    print 'FINISHED READING DATA'
    return [
        (np.array(trainX), np.array(trainY)),
        (np.array(validationX), np.array(validationY)),
        (np.array(testX), np.array(testY))
    ]


def train_nn(train_set, validation_set):

    nn = Regressor(
        layers=[
            Layer("Tanh", units=4),
            Layer("Rectifier", units=4),
            Layer("Sigmoid", units=3),
            Layer("Tanh", units=3),
            Layer("Sigmoid", units=3),
            Layer("Tanh", units=2),
            Layer("Rectifier", units=2),
            Layer("Linear", units=2),
            Layer("Sigmoid")
        ],
        learning_rate=0.001,
        batch_size=20,
        n_iter=1000,
        valid_set=validation_set,
        verbose=True,
    )
    nn.fit(train_set[0], train_set[1])

    return nn


start_time = time.time()
# preprocess_data()
datasets = load_data()
nn = train_nn(datasets[0], datasets[1])
print('score =', nn.score(datasets[2][0], datasets[2][1]))
print time.time() - start_time
