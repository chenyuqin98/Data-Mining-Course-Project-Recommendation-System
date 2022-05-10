# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import numpy as np

from sklearn.feature_selection import RFECV
if __name__ == '__main__':
    support = [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
  True,  True,  True, False,  True,  True,  True,  True,  True,  True,  True, False,
  True,  True, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,
  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
  True,  True,  True,  True,  True,  True,  True,  True,  True,  True]

    X_train = np.random.random((2, 70)).tolist()
    def filter(r):
        rlt = []
        for i in range(len(support)):
            if support[i]:
                rlt.append(r[i])
        return rlt

    X_train = np.array(list(map(lambda x:filter(x), X_train)))
    print(X_train.shape)