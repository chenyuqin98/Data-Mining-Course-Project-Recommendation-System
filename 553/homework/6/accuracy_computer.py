# @Author : Yuqin Chen
# @email : yuqinche@usc.edu


if __name__ == '__main__':
    with open('out.txt') as f:
        guess = f.readlines()[8:]
    guess = list(map(lambda r: r.strip('\n').split(','), guess))
    print(guess[:10])

    with open("hw6_clustering.txt") as in_file:
        ans = in_file.readlines()
    ans = list(map(lambda r: r.strip('\n').split(',')[:2], ans))
    print(ans[:3])