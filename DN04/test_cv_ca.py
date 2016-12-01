import unittest
import random
from sklearn.metrics import roc_auc_score

# To import your solution modify the next row:
from naloga4 import *

class DummyCVLearner:
    """ For CV testing """
    def __call__(self,X,y):
        return DummyCVClassifier(ldata=X)

class YouTestOnTrainingData(Exception): pass
class FoldsNotEqualSize(Exception): pass
class NotAllTested(Exception): pass
class MixedOrder(Exception): pass

class DummyCVClassifier:
    def __init__(self, ldata):
        self.ldata = list(map(list, ldata))
    def __call__(self, x):
        if list(x) in self.ldata:
            raise YouTestOnTrainingData()
        else:
            return [sum(x), len(self.ldata)]

def test_learning(learner, X, y):
    c = learner(X,y)
    results = [ c(x) for x in X ]
    return results

class TestLogisticRegression(unittest.TestCase):

    def data1(self):
        X = numpy.array([[ 5.0, 3.6, 1.4, 0.2 ],
                         [ 5.4, 3.9, 1.7, 0.4 ],
                         [ 4.6, 3.4, 1.4, 0.3 ],
                         [ 5.0, 3.4, 1.5, 0.2 ],
                         [ 5.6, 2.9, 3.6, 1.3 ],
                         [ 6.7, 3.1, 4.4, 1.4 ],
                         [ 5.6, 3.0, 4.5, 1.5 ],
                         [ 5.8, 2.7, 4.1, 1.0 ]])
        y = numpy.array([0, 0, 0, 0, 1, 1, 1, 1])
        return X,y

    def test_ca(self):
        X,y = self.data1()
        self.assertAlmostEqual(CA(y, [[1,0]]*len(y)), 0.5, msg='CA test1')
        self.assertAlmostEqual(CA(y, [[0.5,1]]*len(y)), 0.5, msg='CA test2')
        self.assertAlmostEqual(CA(y, [[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0]]), 0.0, msg='CA test3')

    def test_logreg_noreg_learning_ca(self):
        X,y = self.data1()
        logreg = LogRegLearner(lambda_ = 0)
        pred = test_learning(logreg, X, y)
        ca = CA(y, pred)
        self.assertAlmostEqual(ca, 1.)

    def test_cv(self):
        X,y = self.data1()

        pred = test_cv(DummyCVLearner(),X,y,k=4)
        if len(set([a for _,a in pred])) != 1:
            raise FoldsNotEqualSize()

        signatures = [a for a,_ in pred]
        if len(set(signatures)) != len(y):
            raise NotAllTested()
    
        if signatures != list(map(lambda x: sum(list(x)), X)):
            raise MixedOrder()

    def test_auc(self):
        y_real = [random.randint(0,1) for i in range(20)]
        y_pred_my = [[0, random.uniform(0, 1)] for _ in range(20)]
        y_pred_builtin = [x[1] for x in y_pred_my]
        y_pred_my = [[1-x[1],x[1]] for x in y_pred_my]

        self.assertAlmostEqual(float(roc_auc_score(y_real, y_pred_builtin)), AUC(y_real, y_pred_my), places=10)

if __name__ == '__main__':
    unittest.main()
