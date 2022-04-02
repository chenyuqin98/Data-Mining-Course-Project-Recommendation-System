# calculate RMSE
import numpy as np


def count(file):
    with open(file) as in_file:
        guess = in_file.readlines()[1:]
    with open("data/yelp_val.csv") as in_file:
        ans = in_file.readlines()[1:]

    RMSE = 0
    for i in range(len(guess)):
        diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
        RMSE += diff ** 2
    RMSE = (RMSE / len(guess)) ** (1 / 2)
    print("RMSE: " + str(RMSE))

    prediction = np.array([float(gg.split(',')[2]) for gg in guess])
    print("Prediction mean: " + str(prediction.mean()))
    print("Prediction std:" + str(prediction.std()))
    ground = np.array([float(gg.split(',')[2]) for gg in ans])
    print("Answer mean: " + str(ground.mean()))
    print("Answer std: " + str(ground.std()))


if __name__ == '__main__':
    # with open("test/task2_3output.csv") as in_file:
    # with open("task2_3output.csv") as in_file:
    # print("-----test/task23out.csv-----")
    # count("test/task23out.csv")
    # print("-----task2_2output.csv-----")
    # count("task2_2output.csv")

    count('help/task2_2.csv')