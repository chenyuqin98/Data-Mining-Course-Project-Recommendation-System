"""
Calculate precision and recall
"""

if __name__=='__main__':
    with open("data/pure_jaccard_similarity.csv") as in_file:
        answer = in_file.read().splitlines(True)[1:]
    answer_set = set()
    for line in answer:
        row = line.split(',')
        answer_set.add((row[0], row[1]))

    with open("task1_output.csv") as in_file:
    # with open("task1_test_output.csv") as in_file:
        estimate = in_file.read().splitlines(True)[1:]
    estimate_set = set()
    for line in estimate:
        row = line.split(',')
        estimate_set.add((row[0], row[1]))
        # estimate_set.add(tuple((sorted([row[0], row[1]]))))

    print("Precision:")
    print(len(answer_set.intersection(estimate_set))/len(estimate_set))
    print("Recall:")
    print(len(answer_set.intersection(estimate_set))/len(answer_set))
    print(answer_set.difference(estimate_set))
    print(estimate_set.difference(answer_set))
