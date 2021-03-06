import pickle
from collections import defaultdict
from skgmm import GMMSet
from features import get_feature
import time

class ModelInterface:

    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()

    def enroll(self, name, fs, signal):
        feat = get_feature(fs, signal)
        self.features[name].extend(feat)

    def train_full(self):
        self.gmmset = GMMSet()
        start_time = time.time()
        print('==============',len(self.features.items()))
        for name, feats in self.features.items():
            try:
                self.gmmset.fit_new(feats, name)
            except Exception as e :
                print ("%s failed"%(name))
        print (time.time() - start_time, " seconds")


    def train_single(self,lable):
        start_time = time.time()
        try:
            self.gmmset.fit_new(self.features.__getitem__(lable), lable)
        except Exception as e :
            print ("%s failed"%(lable))
        print (time.time() - start_time, " seconds")

    def dump(self, fname):
        """ dump all models to file"""
        self.gmmset.before_pickle()
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    def predict(self, fs, signal):
        """
        return a label (name)
        """
        try:
            feat = get_feature(fs, signal)
        except Exception as e:
            print (e)
        return self.gmmset.predict_one(feat)

    def verify(self, fs, signal, personid):
        """
        return a label (name)
        """
        try:
            feat = get_feature(fs, signal)
        except Exception as e:
            print (e)
        return self.gmmset.verify(feat, personid)

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R
