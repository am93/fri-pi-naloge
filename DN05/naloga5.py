from collections import defaultdict
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"


def rmse(actual, predict):
    return sqrt(mean_squared_error(actual, predict))


def take_non_zeros(vec1, vec2):
    to_delete = []
    for i, v in enumerate(zip(vec1, vec2)):
        if v[0] == 0 and v[1] == 0:
            to_delete.append(i)
    np.delete(vec1, to_delete)
    np.delete(vec2, to_delete)
    return vec1.reshape(1, -1), vec2.reshape(1, -1)


def average_user_scores(users, user_idxs):
    u_avgs = np.zeros(len(users))
    for user in users.keys():
        u_avgs[user_idxs[user]] = np.average(users[user])
    return u_avgs


def create_user_movie_matrix(data, users, movies):
    matrix = np.zeros((len(users), len(movies)))
    user_idxs = {}
    movies_idxs = {}
    for i, u in enumerate(users.keys()):
        user_idxs[u] = i
        for j, m in enumerate(movies.keys()):
            if i == 0: movies_idxs[m] = j
            matrix[i, j] = data[(u,m)]

    return matrix, user_idxs, movies_idxs


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


def task2_rs2_cosine(matrix, u_idxs, m_idxs, u_avgs):
    predict = []
    actual = []
    with open('movielens-100k-test.tab') as test_file:
        for line in test_file:
            u, m, r = line.rstrip().split('\t')
            actual.append(int(r))
            # movie not present, take user average
            if m not in m_idxs.keys():
                predict.append(u_avgs[u_idxs[user]])
                continue

            # otherwise check which movies were rated by user
            rated_idxs = np.where(matrix[u_idxs[u], :] != 0)[0]    # get user row and find nonzero entries

            # subtract user average scores but keep zeros (unrated)
            crr_vec = matrix[:, m_idxs[m]] - u_avgs
            for x in np.where(matrix[:, m_idxs[m]] == 0)[0]:
                crr_vec[x] = 0

            s1, s2 = [], []
            for rated in rated_idxs:
                # subtract user average scores but keep zeros (unrated)
                tmp_vec = matrix[:,rated] - u_avgs
                for x in np.where(matrix[:, rated] == 0)[0]:
                    tmp_vec[x] = 0
                # calculate cosine similarity, but only on entries where we have both scores
                sim = cosine_similarity(*take_non_zeros(crr_vec, tmp_vec))
                if sim >= 0.0:
                    s1.append(sim * matrix[u_idxs[u], rated])
                    s2.append(sim)

            predict.append(np.sum(s1) / np.sum(s2))
            #if(abs(predict[-1] - actual[-1]) > 1.5):
            #    print(predict[-1],' ',len(s1),' ',np.sum(s1),' ',np.sum(s2))

    return actual, predict


if __name__ == '__main__':
    users = defaultdict(list)
    movies = defaultdict(list)
    data = defaultdict(int)
    train_all = []

    with open('movielens-100k-train.tab') as train_file:
        for line in train_file:
            user, movie, rating = line.rstrip().split('\t')
            train_all.append(int(rating))
            movies[movie].append(int(rating))
            users[user].append(int(rating))
            data[(user, movie)] = int(rating)

    # Task1
    #rmse1 = rmse(*task1_rs_average(users, movies, np.average(train_all)))
    #print(rmse1)

    # Task2
    matrix, u_idxs, m_idxs = create_user_movie_matrix(data, users, movies)
    u_avgs = average_user_scores(users, u_idxs)
    rmse2 = rmse(*task2_rs2_cosine(matrix, u_idxs, m_idxs, u_avgs))
    print(rmse2)