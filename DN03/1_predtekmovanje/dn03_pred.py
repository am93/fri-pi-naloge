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
from sklearn.metrics import mean_absolute_error # TODO dodaj knjiznico

DEP_IDX = -3
ARR_IDX = -1
DRV_IDX =  1
BUS_IDX = 0

HOLIDAYS = ['2012-01-01', '2012-01-02', '2012-02-08', '2012-04-09', '2012-04-27', '2012-05-01', '2012-05-02',
                '2012-05-31', '2012-06-25', '2012-08-15', '2012-10-31', '2012-11-01', '2012-12-25', '2012-12-26']
SCHOOL_HOL = ['2012-02-20', '2012-02-21', '2012-02-22', '2012-02-23', '2012-02-24', '2012-04-30',
              '2012-12-24', '2012-12-25', '2012-12-26', '2012-12-27', '2012-12-28', '2012-12-31']
SUMMER_HOL = ['2012-06-25 00:00:00.000', '2012-08-31 00:00:00.000']

drivers = {}
buses = {}


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
    result = np.zeros(7)
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
    result[5] = driver_average(row[DRV_IDX])
    result[6] = bus_average(row[BUS_IDX])

    return result


def driver_average(driver_id=None, data=None):
    global drivers
    if driver_id is None:
        for row in data:
            if row[DRV_IDX] not in drivers.keys():
                drivers[row[DRV_IDX]] = [lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX]), 1]
            else:
                tmp = drivers[row[DRV_IDX]]
                tmp[0] = lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX])
                tmp[1] += 1
                drivers[row[DRV_IDX]] = tmp
        tmp = {driver : float(drivers[driver][0]) / float(drivers[driver][1])  for driver in drivers.keys()}
        drivers = {driver: tmp[driver] / max(tmp.values()) for driver in tmp}
    else:
        try:
            return drivers[driver_id]
        except KeyError:
            print(sum(drivers.values()) / float(len(drivers)))
            return sum(drivers.values()) / float(len(drivers))


def bus_average(bus_id=None, data=None):
    global buses
    if bus_id is None:
        for row in data:
            if row[BUS_IDX] not in buses.keys():
                buses[row[BUS_IDX]] = [lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX]), 1]
            else:
                tmp = buses[row[BUS_IDX]]
                tmp[0] = lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX])
                tmp[1] += 1
                buses[row[BUS_IDX]] = tmp
        tmp = {bus : float(buses[bus][0]) / float(buses[bus][1])  for bus in buses.keys()}
        buses = {bus: tmp[bus] / max(tmp.values()) for bus in tmp}
    else:
        try:
            return buses[bus_id]
        except KeyError:
            print(sum(buses.values()) / float(len(buses)))
            return sum(buses.values()) / float(len(buses))



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
    lamb = numpy.arange(0.3, 0.6, 0.1)

    for l_idx in range(len(lamb)):
        errors = []
        for idx in range(len(new_data)):
            train_data = [d for i in range(len(new_data)) if i != idx for d in new_data[i]] # flatten array
            test_data = new_data[idx]
            lin_l = LinearLearner(lamb[l_idx])
            lin_c = lin_l(train_data)

            predict = []
            real = []
            for elem in test_data:
                predict.append(lin_c(elem))
                real.append(lpputils.tsdiff(elem[ARR_IDX], elem[DEP_IDX]))
            errors.append(mean_absolute_error(real,predict))
        print("Lambda : {0} , Errors: {1}, Predict: {2}".format(lamb[l_idx], np.mean(errors), predict))


def precomp_prediction(data_train, data_test, lamb=0.3):
    lin_l = LinearLearner(lamb)
    lin_c = lin_l(data_train)

    out_file = open("precomp_results4.txt", "wt")
    for row in data_test:
        out_file.write(lpputils.tsadd(row[DEP_IDX], lin_c(row)) + "\n")
    out_file.close()


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

if __name__ == "__main__":
    print("Testiranje za predtekomvanje ...")
    data_train = read_file("train_pred.csv.gz")
    data_test = read_file("test_pred.csv.gz")
    driver_average(None, data_train)
    print(" -- izracunal povprecje voznikov ...")
    bus_average(None, data_train)
    print(" -- izracunal povprecje busov ...")
    cross_validate(data_train,3,False)
    #precomp_prediction(data_train,data_test, 0.3)
    print(" -- koncano")



