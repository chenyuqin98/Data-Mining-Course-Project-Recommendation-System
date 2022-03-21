# calculate RMSE
import numpy as np

if __name__ == '__main__':
    # with open("test/task2_2output.csv") as in_file:
    # with open("task2_3output.csv") as in_file:
    with open("data/yelp_val.csv") as in_file:
        guess = in_file.readlines()[1:]
    with open("data/yelp_val.csv") as in_file:
        ans = in_file.readlines()[1:]
    res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_guess = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_ans = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    large_small = {"large": 0, "small": 0}

    RMSE = 0
    for i in range(len(guess)):
        # if float(guess[i].split(",")[2]) < 1:
        #     dist_guess["<1"] = dist_guess["<1"] + 1
        # elif 2 > float(guess[i].split(",")[2]) >= 1:
        #     dist_guess["1~2"] = dist_guess["1~2"] + 1
        # elif 3 > float(guess[i].split(",")[2]) >= 2:
        #     dist_guess["2~3"] = dist_guess["2~3"] + 1
        # elif 4 > float(guess[i].split(",")[2]) >= 3:
        #     dist_guess["3~4"] = dist_guess["3~4"] + 1
        # else:
        #     dist_guess["4~5"] = dist_guess["4~5"] + 1
        #
        # if float(ans[i].split(",")[2]) == 1:
        #     dist_ans["1"] = dist_ans["1"] + 1
        # elif float(ans[i].split(",")[2]) == 2:
        #     dist_ans["2"] = dist_ans["2"] + 1
        # elif float(ans[i].split(",")[2]) == 3:
        #     dist_ans["3"] = dist_ans["3"] + 1
        # elif float(ans[i].split(",")[2]) == 4:
        #     dist_ans["4"] = dist_ans["4"] + 1
        # else:
        #     dist_ans["5"] = dist_ans["5"] + 1
        diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
        RMSE += diff**2
        if abs(diff) < 1:
            res["<1"] = res["<1"] + 1
        elif 2 > abs(diff) >= 1:
            res["1~2"] = res["1~2"] + 1
            # print(guess[i].split(","))
            # print(ans[i].split(","))
            # print("========")
            # if diff > 0:
            #     large_small["large"] = large_small["large"] + 1
            # else:
            #     large_small["small"] = large_small["small"] + 1
        elif 3 > abs(diff) >= 2:
            res["2~3"] = res["2~3"] + 1
            # if diff > 0:
            #     large_small["large"] = large_small["large"] + 1
            # else:
            #     large_small["small"] = large_small["small"] + 1
            # print(guess[i].split(","))
            # print(ans[i].split(","))
            # print("========")
        elif 4 > abs(diff) >= 3:
            res["3~4"] = res["3~4"] + 1
        else:
            res["4~5"] = res["4~5"] + 1
    RMSE = (RMSE/len(guess))**(1/2)
    print("RMSE: "+str(RMSE))
    prediction = np.array([float(gg.split(',')[2]) for gg in guess])
    print("Prediction mean: " + str(prediction.mean()))
    print("Prediction std:" + str(prediction.std()))
    ground = np.array([float(gg.split(',')[2]) for gg in ans])
    print("Answer mean: "+str(ground.mean()))
    print("Answer std: "+str(ground.std()))
    # print(res)
    # print(dist_ans)
    # print(dist_guess)
    # print(large_small)