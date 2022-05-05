# @Author : Yuqin Chen
# @email : yuqinche@usc.edu


def compute_RMSE(pred, truth):
    return (float(pred) - float(truth)) ** 2


if __name__ == '__main__':
    content_base_better , total = 0, 0
    with open("data/yelp_val.csv") as val_file:
        ans = val_file.readlines()[1:]
    with open('2_models_results.csv') as f:
        rlts = f.readlines()
        for i in range(len(rlts)):
            content_base_RMSE = compute_RMSE(rlts[i].split(',')[0], ans[i].split(",")[2])
            xgboost_RMSE = compute_RMSE(rlts[i].split(',')[1], ans[i].split(",")[2])
            if content_base_RMSE < xgboost_RMSE:
                content_base_better += 1
    total = len(rlts)
    print(content_base_better, total)