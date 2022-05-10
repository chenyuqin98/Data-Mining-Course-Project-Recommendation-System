# @Author : Yuqin Chen
# @email : yuqinche@usc.edu
import numpy as np


def compute_RMSE(pred, truth):
    return (float(pred) - float(truth)) ** 2


if __name__ == '__main__':
    content_base_better , total = 0, 0
    with open("../data/yelp_val.csv") as val_file:
        ans = val_file.readlines()[1:]

    better_model_test_Y = [0] * len(ans)
    content_base_RMSE, xgboost_RMSE = [0] * len(ans), [0] * len(ans)
    with open('../2_models_results.csv') as f:
        pred = f.readlines()
        for i in range(len(pred)):
            content_base_RMSE[i] = compute_RMSE(pred[i].split(',')[0], ans[i].split(",")[2])
            xgboost_RMSE[i] = compute_RMSE(pred[i].split(',')[1], ans[i].split(",")[2])
            if content_base_RMSE[i] < xgboost_RMSE[i]:
                content_base_better += 1
                better_model_test_Y[i] = 1
    np.save('better_model_test_Y.npy', np.array(better_model_test_Y))

    total = len(pred)
    print(content_base_better, total)
    
    with open('analyse_rlt.csv', 'w+') as f:
        # f.writelines('CF_prediction, Y_pred, a' + '\n')
        for i in range(len(content_base_RMSE)):
            f.writelines(str(content_base_RMSE[i])+','+str(xgboost_RMSE[i])+','+str(pred[i].split(',')[2])+'\n')