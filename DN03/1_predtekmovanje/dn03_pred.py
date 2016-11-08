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

DEP_IDX = -3
ARR_IDX = -1

HOLIDAYS = ['2012-01-01', '2012-01-02', '2012-02-08', '2012-04-09', '2012-04-27', '2012-05-01', '2012-05-02',
                '2012-05-31', '2012-06-25', '2012-08-15', '2012-10-31', '2012-11-01', '2012-12-25', '2012-12-26']
SCHOOL_HOL = ['2012-02-20', '2012-02-21', '2012-02-22', '2012-02-23', '2012-02-24', '2012-04-30']
SUMMER_HOL = ['2012-06-25', '2012-08-31']


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
    result = np.zeros(5)
    result[0] = lpputils.parsedate(row[DEP_IDX]).weekday() / 7.0 # day
    result[1] =  lpputils.parsedate(row[DEP_IDX]).hour / 24.0 # hour
    date = lpputils.parsedate(row[DEP_IDX]).date()

    holiday = 0
    school_hol = 0
    summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[2] = holiday
    result[3] = school_hol
    result[4] = summer_hol

    return result


def read_file(filename):
    """
    Read file and store lines into array for later processing.
    """
    f = gzip.open(filename, "rt")
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
    TODO
    """
    new_data = cv_data_split(data, k, cons) # split data
    lamb = numpy.arange(0.03, 0.3, 0.03)

    for l_idx in range(len(lamb)):
        errors = []
        for idx in range(len(new_data)):
            train_data = [d for i in range(len(new_data)) if i != idx for d in new_data[i]] # flatten array
            test_data = new_data[idx]
            lin_l = LinearLearner(lamb[l_idx])
            lin_c = lin_l(train_data)

            # TODO - nadaljuj implementacijo



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
        x = numpy.hstack(([1.], x))
        return hl(x, self.th)

if __name__ == "__main__":
    print("Testiranje za predtekomvanje ...")
    data = read_file("train_pred.csv.gz")
    abc = list(range(12))
    cross_validate(abc,3, False)



