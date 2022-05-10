# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import xgboost
import numpy as np

if __name__ == '__main__':
    params = {'max_depth': 5, 'objective': 'multi:softprob', 'num_class': 2}
    combine_model_train_X = 455854*[19, 0.503313395229126, 0.528621536705927, 0.5324246384982259, 0.567944534924714, 0.573612572929557]
    print('combine_model_train_X shape: ', np.shape(combine_model_train_X))
    combine_model_train_X = np.array(combine_model_train_X).reshape((455854, -1))
    print('combine_model_train_X shape: ', np.shape(combine_model_train_X))
    better_model = 455854*[1]
    # data = np.random.rand(5, 10)
    # label = np.random.randint(2, size=5)
    # print(label, np.shape(label))
    # dtrain = xgboost.DMatrix(data, label=label)
    dtrain = xgboost.DMatrix(combine_model_train_X, better_model)
    combine_model = xgboost.train(params, dtrain)
    # print(combine_model.predict(xgboost.DMatrix(data)))
    print(combine_model.predict(xgboost.DMatrix(combine_model_train_X)))
    # better_model_pred = combine_model.predict(combine_model_pred_X)