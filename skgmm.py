from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
import operator
import numpy as np
import time

class GMMSet:

    def __init__(self, gmm_order = 32):
        self.gmms = []
        self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GaussianMixture(self.gmm_order)
        gmm.fit(x)

        gmmvalue = []
        gmmvalue.append(gmm)
        gmmvalue.append(label)
        self.gmms.append(gmmvalue)
        print(self.gmms)

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    def predict_one(self, x):
        # test tremendous amount
        for i in range(1):
            #start_time = time.time()

            #取GMMset的第一列，只留频谱模型数据
            self.gmms = [gmm_model[0] for gmm_model in self.gmms]

            scores = [self.gmm_score(gmm, x) for gmm in self.gmms]
            print(scores)
            #end_time = time.time()
            #print('耗时：',end_time-start_time)
            p = sorted(enumerate(scores), key=operator.itemgetter(1), reverse=True)
            p = [(str(self.y[i]), y, p[0][1] - y) for i, y in p]
            result = [(self.y[index], value) for (index, value) in enumerate(scores)]
            print(result)
            p = max(result, key=operator.itemgetter(1))
            print(p)
        return p[0],p[1]

    def verify(self, x, personid):
        #获取模型中personid对应的频谱数据
        gmm_label = [gmm[1] for gmm in self.gmms]
        gmm = self.gmms[gmm_label.index(personid)][0]
        score = self.gmm_score(gmm, x)



        return score

    def before_pickle(self):
        pass

    def after_pickle(self):
        pass
