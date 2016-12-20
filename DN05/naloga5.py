from collections import defaultdict
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"


def rmse(actual, predict):
    return sqrt(mean_squared_error(actual, predict))


def task1_rs_average(users, movies, train_avg, correction=True):
    """
    Implementation of first task in homework. Recommendation is based on movie quality,
    which is evaluated on average movie rating, and generosity of user. If movie or user
    is not present in train set, we take an average of all training entries as predicition.
    Predicition is evaluated as : (user_pred - train_avg) + (movie_pred - train_avg) + train_avg
    """
    actual, predict = [], []
    with open('movielens-100k-test.tab') as test_file:
        for line in test_file:
            u, m, r = line.rstrip().split('\t')

            # get user prediction
            if u not in users:
                u_pred = train_avg
            else:
                u_pred = np.average(users[u])

            # get movie prediction
            if m not in movies:
                m_pred = train_avg
            else:
                m_pred = np.average(movies[m])

            # make a prediction
            if correction:
                predict.append(min(max(u_pred + m_pred - train_avg, 1), 5))
            else:
                predict.append(u_pred + m_pred - train_avg)
            actual.append(float(r))

    return actual, predict

if __name__ == '__main__':
    users = defaultdict(list)
    movies = defaultdict(list)
    train_all = []

    with open('movielens-100k-train.tab') as train_file:
        for line in train_file:
            user, movie, rating = line.rstrip().split('\t')
            train_all.append(int(rating))
            movies[movie].append(int(rating))
            users[user].append(int(rating))

    # Task1
    rmse1 = rmse(*task1_rs_average(users, movies, np.average(train_all), False))
    print(rmse1)
