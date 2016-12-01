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


def test_cv(learner, X, y, k=5):
    """
    Implementation of k-fold cross validation, which returns predictions in the same order as in X. Function
    never uses same examples for training and prediction.
    """
    x_splits = numpy.array_split(X, k)
    y_splits = numpy.array_split(y, k)

    predictions = []

    for crr_x, crr_y in zip(x_splits, y_splits):
        train_x = numpy.vstack([x for x in x_splits if (x - crr_x).any()]) # current split is only for prediction
        train_y = numpy.hstack([y for y in y_splits if (y - crr_y).any()]) # current split is only for prediciton
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
    #
    # Usage example
    #

    X,y = load('reg.data')
    X_img, y_img = load('slike_znacilke.csv')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X,y) # we get a model

    prediction = classifier(X[0]) # prediction for the first training example
    print(prediction)

    # check all predicitions from example data (without regularization)
    #for (i,ex) in enumerate(X):
    #    pred, prob = max(enumerate(classifier(ex)), key=operator.itemgetter(1))
    #    print("Pred:{0}, Real:{2}, Prob:{1}".format(pred, prob, y[i]))

    # draw
    # TODO: test regularization with different lambda values (report three most interesting images)
    #draw_decision(X, y, classifier, 0, 1)

    # AUC test
    # DONE: vrne iste rezultate kot sklearn.metrics.roc_auc_score
    y_real1 = [1,1,1,0,0,0]
    y_pred1 = [[0, 1], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [1.0, 0.0]]
    y_real2 = [1,1,0,1,0,0]
    y_pred2 = [[0, 1], [0.1, 0.9], [0.4, 0.6], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]]
    print(AUC(y_real1,y_pred1))
    print(AUC(y_real2,y_pred2))

