import numpy
from scipy.optimize import fmin_l_bfgs_b
import operator
from draw import draw_decision

__author__ = "Anze Medved, 63120191"
__email__ = "am1947@student.uni-lj.si"
__course__ = "Poslovna inteligenca"

def load(name):
    """
    Open the file. Return a data matrix X (columns are features)
    and a vector of classes.
    """
    data = numpy.loadtxt(name)
    X, y = data[:,:-1], data[:,-1].astype(numpy.int)
    return X,y


def h(x, theta):
    """ 
    Predict the probability for class 1 for the current instance
    and a vector theta.
    """
    return 1.0 / (1 + numpy.exp(-x.dot(theta)))


def cost(theta, X, y, lambda_):
    """
    Return the value of the cost function. Because the optimization algorithm
    used can only do minimization you will have to slightly adapt equations from
    the lectures.
    """
    regularize = lambda_ / (2 * X.shape[0]) * numpy.sum(numpy.square(theta))
    return (-1.0 / X.shape[0]) * numpy.sum(y * numpy.log(h(X,theta)) + (1 - y) * numpy.log(1 - h(X,theta))) + regularize


def grad(theta, X, y, lambda_):
    """
    The gradient of the cost function. Return a numpy vector of the same
    size at theta.
    """
    regularize = theta * (lambda_ / X.shape[0])
    return 1.0 / X.shape[0] * (h(X,theta) - y).dot(X) + regularize


def test_learning(learner, X, y):
    """
    Incorrect approach - here just for demonstration
    """
    c = learner(X,y)
    results = [ c(x) for x in X ]
    return results


def test_cv(learner, X, y, k=5):
    """
    Implementation of k-fold cross validation, which returns predictions in the same order as in X. Function
    never uses same examples for training and prediction.
    """
    x_splits = numpy.array_split(X, k)
    y_splits = numpy.array_split(y, k)

    predictions = []

    for crr_i, (crr_x, crr_y) in enumerate(zip(x_splits, y_splits)):
        train_x = numpy.vstack([x for i,x in enumerate(x_splits) if i != crr_i]) # current split is only for prediction
        train_y = numpy.hstack([y for i,y in enumerate(y_splits) if i != crr_i]) # current split is only for prediciton
        predictor = learner(train_x, train_y)
        predictions += [predictor(crr_x[i]) for i in range(len(crr_y))]

    return predictions


def CA(real, predictions):
    """
    Function computes classification accuracy.
    """
    result = 0.0
    for crr_r, crr_p in zip(real, predictions):
        idx, _ = max(enumerate(crr_p), key=operator.itemgetter(1))
        if idx == crr_r :
            result += 1

    return result / len(predictions)

def AUC(real, predicitions):
    """
    Function computes area under curve. Real is a list of "1" and "0" and represent real class.
    Predicitions is list of sublist, where each sublist contains probability for class "0" and "1".
    """
    class_prob = [x for [_,x] in predicitions]
    zipped = sorted(zip(real, class_prob), key=operator.itemgetter(1), reverse=True)
    pos = [p[1] for p in zipped if p[0] == 1]
    neg = [n[1] for n in zipped if n[0] == 0]

    if len(pos) == 0 or len(neg) == 0:
        print("AUC not defined if only single class is present !")
        return -1

    return 1.0 / (len(pos) * len(neg)) * sum([1 for p in pos for n in neg if p > n])


def test_model_regularization(X,y, k=5):
    """
    Function tests our model with different values of lambda for regularization and reports CA and AUC
    for incorrect and correct approach (cross-validation).
    """
    lambdas = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1, 3, 5, 10, 20, 50, 100, 500]
    for l in lambdas:
        print("Testing with lambda = {0}".format(l))
        learner = LogRegLearner(lambda_=l)

        # make predicitions
        learn_pred = test_learning(learner, X, y)
        cv_pred = test_cv(learner, X,y, k)

        # evaluate predicitions
        ca_learn = CA(y, learn_pred)
        ca_cv = CA(y, cv_pred)
        auc_learn = AUC(y, learn_pred)
        auc_cv = AUC(y, cv_pred)

        # report
        print("--> Predictions on learning - CA: {0}, AUC: {1}".format(ca_learn,auc_learn))
        print("--> 5-fold cross validation - CA: {0}, AUC: {1}".format(ca_cv, auc_cv))



class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Predict the class for a vector of feature values.
        Return a list of [ class_0_probability, class_1_probability ].
        """
        x = numpy.hstack(([1.], x))
        p1 = h(x, self.th)
        return [ 1-p1, p1 ] 


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Build a prediction model for date X with classes y.
        """
        X = numpy.hstack((numpy.ones((len(X),1)), X))

        #optimization as minimization
        theta = fmin_l_bfgs_b(cost,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)

if __name__ == "__main__":
    # load data
    X,y = load('reg.data')
    X_img, y_img = load('slike_znacilke.csv')

    test_model_regularization(X,y)

    # check all predicitions from example data (without regularization)
    #learner = LogRegLearner(lambda_=0.)
    #classifier = learner(X_img, y_img)  # we get a model
    #for (i,ex) in enumerate(X):
    #    pred, prob = max(enumerate(classifier(ex)), key=operator.itemgetter(1))
    #    print("Pred:{0}, Real:{2}, Prob:{1}".format(pred, prob, y[i]))

    # draw
    #draw_decision(X, y, classifier, 0, 1)

