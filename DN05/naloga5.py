from collections import defaultdict
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
from matplotlib import pyplot as plt
from random import random
import matplotlib.cm as cmx
import matplotlib.colors as colors

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"


def random_color():
    return (random(), random(), random())


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


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
    """
    Implementation of second task in homework. Recomendation is based on movie similarity, with cosine
    similarity used as measure.
    """
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


def matrix_factorization(data, user_idxs, mov_idxs, iters=300, k=2, _lambda=0.001, _eps=0.03):
    """
    Implementation of matrix factorization. Parameter lambda represents learning rate and parameter
    eps degree of regularization. Matrices P and Q are initialized to some small random values.
    Than we iterate multiple times through train set and use gradient descent to minimize errors
    between original matrix R and P*Q.
    """
    P = np.random.uniform(-0.01, 0.01, [len(user_idxs.keys()), k])
    Q = np.random.uniform(-0.01, 0.01, [k, len(mov_idxs.keys())])

    for _ in range(iters):
        total_err = []
        with open('movielens-100k-train.tab', encoding = "ISO-8859-1") as train_file:
            for line in train_file:
                u, m, r = line.rstrip().split('\t')
                u_idx = user_idxs[u]
                m_idx = mov_idxs[m]
                err = data[(u, m)] - Q.T[m_idx, :].dot(P[u_idx, :])
                total_err += [err]
                for x in range(k):
                    P[u_idx][x] += _lambda* (err * Q[x][m_idx] - _eps * P[u_idx][x])
                    Q[x][m_idx] += _lambda * (err * P[u_idx][x] - _eps * Q[x][m_idx])

        #print(np.average(total_err))

    return P, Q


def task3_rs3_rismf(u_avgs, u_idxs, m_idxs, P, Q):
    predict, actual = [], []

    with open('movielens-100k-test.tab', encoding='ISO-8859-1') as test_file:
        for line in test_file:
            u, m, r = line.rstrip().split('\t')
            actual.append(int(r))
            # movie not present, take user average
            if m not in m_idxs.keys():
                predict.append(u_avgs[u_idxs[user]])
                continue

            u_idx = u_idxs[u]
            m_idx = m_idxs[m]

            # make a predicition (keep it in range 1-5)
            pr = min(5, max(0, Q.T[m_idx].dot(P[u_idx])))
            predict.append(pr)

    return actual, predict


def visualize(Q, titles):
    x_cord, y_cord = [], []
    for t in titles:
        x_cord.append(Q[0][m_idxs[t]])
        y_cord.append(Q[1][m_idxs[t]])

    cnt = 0
    for x, y in zip(x_cord, y_cord):
        plt.plot(x, y, c=random_color(), marker='o', label=str(cnt) + ' ' + titles[cnt])  # plot dots
        cnt += 1
    plt.tight_layout()
    for i, text in enumerate(titles):
        plt.annotate(i, (x_cord[i], y_cord[i]))

    plt.legend(loc='best', ncol=2)
    plt.show()


if __name__ == '__main__':
    users = defaultdict(list)
    movies = defaultdict(list)
    data = defaultdict(int)
    train_all = []

    with open('movielens-100k-train.tab', encoding = "ISO-8859-1") as train_file:
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
    #rmse2 = rmse(*task2_rs2_cosine(matrix, u_idxs, m_idxs, u_avgs))
    #print(rmse2)

    # Task3
    for k in range(7,11):
        P, Q = matrix_factorization(data, u_idxs, m_idxs, k=k, iters=250)
        rmse3 = rmse(*task3_rs3_rismf(u_avgs, u_idxs, m_idxs, P, Q))
        print(k,' -> ',rmse3)

    # Task4
    titles = ['Shawshank Redemption, The (1994)', 'Star Wars (1977)', 'Pulp Fiction (1994)',
              'Stargate (1994)', 'Home Alone 3 (1997)', 'Home Alone (1990)',
              'Return of the Pink Panther, The (1974)', 'Good, The Bad and The Ugly, The (1966)',
              'Return of the Jedi (1983)', 'Empire Strikes Back, The (1980)',
              'Star Trek: First Contact (1996)', 'Star Trek VI: The Undiscovered Country (1991)',
              'Star Trek: The Wrath of Khan (1982)', 'Star Trek III: The Search for Spock (1984)',
              'Star Trek IV: The Voyage Home (1986)', 'Grumpier Old Men (1995)', 'Dumb & Dumber (1994)',
              'Jumanji (1995)', 'Winnie the Pooh and the Blustery Day (1968)', 'Toy Story (1995)']
    #visualize(Q, titles)
