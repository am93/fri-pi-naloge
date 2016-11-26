from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse
import gzip, csv
import scipy.sparse as sp
import numpy as np
import lpputils
from random import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import model_helper as mh

MODEL_NAME = "MODEL9"
DEP_IDX = -3
ARR_IDX = -1
DRV_IDX =  1
BUS_IDX = 0

def append_ones(X):
    if sp.issparse(X):
        return sp.hstack((np.ones((X.shape[0], 1)), X)).tocsr()
    else:
        return np.hstack((np.ones((X.shape[0], 1)), X))


def hl(x, theta):
    """
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return x.dot(theta)


def cost_grad_linear(theta, X, y, lambda_):
    #do not regularize the first element
    sx = hl(X, theta)
    j = 0.5*numpy.mean((sx-y)*(sx-y)) + 1/2.*lambda_*theta[1:].dot(theta[1:])/y.shape[0]
    grad = X.T.dot(sx-y)/y.shape[0] + numpy.hstack([[0.],lambda_*theta[1:]])/y.shape[0]
    return j, grad


def parse_row(row):
    """
    Function parses row from data file, to row with attributes.
    """
    return mh.model_getter(MODEL_NAME)(row)


def read_file(filename):
    """
    Read file and store lines into array for later processing.
    """
    f = gzip.open(filename, "rt", encoding="utf8")
    reader = csv.reader(f, delimiter="\t")
    next(reader) # skip header line
    return list(reader)


def cv_data_split(data, k, cons=True):
    """
    Function splits data in k equal subsets. If parameter cons is set to True, then we take consecutive
    rows, otherwise we take random rows.
    """
    subsize = int(len(data) / k)
    if cons:
        new_data = [data[subsize*i:subsize*(i+1)] for i in range(0,k)]
        return new_data
    else:
        new_data = data.copy() # create a deep copy of data
        shuffle(new_data)
        new_data = [new_data[subsize * i:subsize * (i + 1)] for i in range(0, k)]
        return new_data


def cross_validate(data, k=3, cons=True):
    """
    Implementation of cross validation.
    """
    lamb = numpy.arange(0.1, 0.6, 0.1)

    for l_idx in range(len(lamb)):
        errors = []
        new_data = cv_data_split(data, k, cons)  # split data
        for idx in range(len(new_data)):
            train_data = [d for i in range(len(new_data)) if i != idx for d in new_data[i]] # flatten array
            test_data = new_data[idx]
            lin_l = LineSpecificLearner(LinearLearner(lamb[l_idx]))
            lin_c = lin_l(train_data)

            predict = []
            real = []
            for elem in test_data:
                predict.append(lin_c(elem))
                real.append(lpputils.tsdiff(elem[ARR_IDX], elem[DEP_IDX]))
            errors.append(mean_absolute_error(real,predict))
        print("Lambda : {0} , Errors: {1}".format(lamb[l_idx], np.mean(errors)))


def comp_prediction(data_train, data_test, outfile, lamb=0.3):
    #lin_l = LineSpecificLearner(LinearLearner(lamb))
    #lin_c = lin_l(data_train)

    rf_l = LineSpecificLearner(RFRegressorLearner())
    rf_c = rf_l(data_train)

    out_file = open(outfile, "wt")
    for row in data_test:
        out_file.write(lpputils.tsadd(row[DEP_IDX], rf_c(row)) + "\n")
    out_file.close()


def line_identifier(row):
    """
    Creates line identifier based on single row of data.
    Identifier consist of route number, direction, identifier and first station.
    """
    return tuple(row[2:6])


class RFRegressorLearner(object):

    def __init__(self, n_tree=50, crit='mae', threads=-1, depth=3):
        self.regressor = RandomForestRegressor(n_estimators=n_tree, criterion=crit, n_jobs=threads, max_depth=depth)

    def __call__(self, data):
        X, y = [], []
        for row in data:
            X.append(parse_row(row))
            y.append(lpputils.tsdiff(row[ARR_IDX], row[DEP_IDX]))

        X = scipy.sparse.csr_matrix(X)
        y = np.array(y)

        self.regressor.fit(X, y)
        print("Regressor has been fitted !")
        return RFRegressorPredictor(self.regressor)


class RFRegressorPredictor(object):
    def __init__(self, regressor):
        self.regressor = regressor

    def __call__(self, x):
        return self.regressor.predict(x)[0]



class LinearLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, data):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X, y = [], []
        for row in data:
            X.append(parse_row(row))
            y.append(lpputils.tsdiff(row[ARR_IDX], row[DEP_IDX]))

        X = scipy.sparse.csr_matrix(X)
        X = append_ones(X)
        y = np.array(y)

        th = fmin_l_bfgs_b(cost_grad_linear,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_))[0]

        return LinearRegClassifier(th)


class LinearRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = parse_row(x)
        x = numpy.hstack(([1.], x))
        return hl(x, self.th)


class LineSpecificLearner(object):
    def __init__(self, learner):
        self.learner = learner

    def __call__(self, data):
        line_data = defaultdict(list)
        line_class = {}
        # separate input data by bus lines
        for row in data:
            line_data[line_identifier(row)].append(row)

        print("Data has been splitted by lines!")

        # create classifiers for each lane
        for key in line_data.keys():
            print("Working on ", key, " size ",str(len(line_data[key])))
            line_class[key] = self.learner(line_data[key])
        return LineSpecificClassifier(line_class)

class LineSpecificClassifier(object):
    def __init__(self, line_class):
        self.line_class = line_class

    def __call__(self, x):
        # try to use classifier
        try:
            return self.line_class[line_identifier(x)](x)

        except KeyError:
            print("No data for line - {0}".format(line_identifier(x)))
            closest = find_closest_line(line_identifier(x), self.line_class.keys())
            if closest != None:
                print("Found similar - {0}".format(closest))
                return self.line_class[closest](x)# try with closest line
            else:
                return np.mean([c(x) for c in self.line_class.values()]) # exception: new line - take average


def find_closest_line(search_key, all_lines):
    """
    Try to find similar line, with same line number and direction.
    """
    for line_key in all_lines:
        # search for match in line number and direction
        if line_key[0:2] == search_key[0:2]:
            return line_key
    # no match was found
    return None


if __name__ == "__main__":
    print("Testiranje za tekomvanje ...")
    data_train = read_file("train.csv.gz")
    data_test = read_file("test.csv.gz")
    print("Podatki prebrani ...")
    mh.model_init(data_train,MODEL_NAME)
    #cross_validate(data_train,3,False)
    comp_prediction(data_train,data_test, "comp_results15.txt", 3)
    #mh.visualize(data_train,11,1,30)
    #ap.parse_arso_data()
    print(" -- koncano")



