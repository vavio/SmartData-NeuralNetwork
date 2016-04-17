# dataset: https://data.gov.au/dataset/sample-household-electricity-time-of-use-data/resource/ed7aaa03-6282-4254-9dcb-0e80bc6dc90d?inner_span=True
#  Organisations Department of Industry, ... Sample household electricity ... Smart Grid Smart Cities Data ...

from pymongo import MongoClient
from datetime import datetime, timedelta, datetime
from sklearn import preprocessing
from sknn.mlp import Regressor, Layer

import time
import numpy as np
import matplotlib.pyplot as plt


def convert_time(hours, minutes):
    return hours * 60 + minutes


def preprocess_data(input_file='dataset.txt'):
    line_number = 0
    customer_map = {}
    day_subtract = timedelta(days=1)
    week_subtract = timedelta(days=7)
    hour_subtract = timedelta(hours=1)
    twohour_subtract = timedelta(hours=2)
    halfhour_subtract = timedelta(minutes=30)
    year_subtract = timedelta(weeks=51)

    plotX = []
    plotY = []

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

        if customer_id != '8664162':
            continue
        isodate = datetime.strptime(raw_data[1], "%d/%m/%Y %H:%M")
        supply_usage = float(raw_data[2])

        if customer_id not in customer_map:
            print customer_id
            customer_map[customer_id] = {}

        if isodate - year_subtract in customer_map[customer_id]:
            break

        customer_map[customer_id][isodate] = supply_usage

        previous_day_usage = customer_map[customer_id].get(isodate - day_subtract, 0)
        previous_week_usage = customer_map[customer_id].get(isodate - week_subtract, 0)
        previous_hour_usage = customer_map[customer_id].get(isodate - hour_subtract, 0)
        previous_twohour_usage = customer_map[customer_id].get(isodate - twohour_subtract, 0)
        previous_halfhour_usage = customer_map[customer_id].get(isodate - halfhour_subtract, 0)

        plotX.append(isodate.timetuple().tm_yday + convert_time(isodate.hour, isodate.minute) / 1440.0)
        plotY.append(supply_usage)

        data.append([
            convert_time(isodate.hour, isodate.minute),
            isodate.weekday(),
            previous_day_usage,
            previous_week_usage,
            previous_hour_usage,
            previous_twohour_usage,
            previous_halfhour_usage,
            supply_usage])


    print 'PLOTTING DATA...'
    print len(plotX), len(plotY)
    plt.figure(1)
    max_plot = 200

    plt.plot(plotX[:max_plot], plotY[:max_plot], 'bo', plotX[:max_plot], plotY[:max_plot], 'k')
    plt.grid(True)
    plt.title('UID 8664162')
    plt.xlabel('Day of year')
    plt.ylabel('Usage $[kW/h]$')

    plt.show()
    return

    print 'NORMALIZING DATA...'
    data = np.array(data)
    N = data.shape[0]
    X = preprocessing.scale(data[:, :-1]).reshape(N, data.shape[1] - 1)
    Y = np.array(data[:, -1])

    client = MongoClient()
    db = client.smart_grid

    # db_cache = []
    # print 'SAVING RAW DATA TO MONGODB...'
    # for cust_id in customer_map:
    #     print '\tCUSTOMER_ID {0}'.format(cust_id)
    #     for cust_date in customer_map[customer_id]:
    #         db_cache.append({
    #             'customer_id': cust_id,
    #             'datetime': cust_date,
    #             'supply_usage': customer_map[customer_id][cust_date]
    #         })
    #
    #     db.user_info.insert_many(db_cache)
    #     db_cache = []

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
            'previous_day_usage': current_data[2],
            'previous_week_usage': current_data[3],
            'previous_hour_usage': current_data[4],
            'previous_twohour_usage': current_data[5],
            'previous_halfhour_usage': current_data[6],
            'current_usage': Y[i]
        })

    if len(db_cache) != 0:
        db.user_data.insert_many(db_cache)


def get_vector(data):
    return [
        1.0,  # bias
        data['time_of_day'],
        data['week_day'],
        data['previous_day_usage'],
        data['previous_week_usage'],
        data['previous_hour_usage'],
        data['previous_twohour_usage'],
        data['previous_halfhour_usage'],
        data['current_usage'],
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

    count = db.user_data.find({}).count()
    train_size = int(count * 0.70)
    validation_size = int(count * 0.15)

    print count, train_size, validation_size

    for data in db.user_data.find({}):

        if idx % 100000 == 0:
            print '\tIDX {0}'.format(idx)

        idx += 1

        vec = np.array(get_vector(data))
        X = vec[:-1]
        y = vec[-1]

        if train_size > 0:
            trainX.append(X)
            trainY.append(y)
            train_size -= 1
        elif validation_size > 0:
            validationX.append(X)
            validationY.append(y)
            validation_size -= 1
        else:
            testX.append(X)
            testY.append(y)

    print 'FINISHED READING DATA'
    return [
        (np.array(trainX), np.array(trainY)),
        (np.array(validationX), np.array(validationY)),
        (np.array(testX), np.array(testY))
    ]


def train_nn(train_set, validation_set):

    nn = Regressor(
        layers=[
            Layer("Sigmoid", units=2),
            Layer("Sigmoid")
        ],
        learning_rate=0.0001,
        batch_size=5,
        n_iter=10000,
        valid_set=validation_set,
        verbose=True,
    )
    nn.fit(train_set[0], train_set[1])

    return nn


def mean_absolute_percentage_error(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return np.mean(np.abs(ypred[idx]-ytrue[idx])/ytrue[idx])


start_time = time.time()
preprocess_data()
datasets = load_data()
nn = train_nn(datasets[0], datasets[1])
y_pred = nn.predict(datasets[2][0])

print 'mape\t= {0}'.format(mean_absolute_percentage_error(y_pred, datasets[2][1]))
print 'score\t= {0}'.format(nn.score(datasets[2][0], datasets[2][1]))
print time.time() - start_time
