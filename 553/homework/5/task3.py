from blackbox import BlackBox
import random
import time
import sys

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

# s: memory capacity, n: n-th user
s = 100
n = 0


if __name__=='__main__':
    start_time = time.time()
    bx = BlackBox()
    random.seed(553)

    with open(output_filename, 'w+') as f:
        f.writelines("seqnum,0_id,20_id,40_id,60_id,80_id\n")

    user_in_memory = []
    for step in range(num_of_asks):
        stream_user = bx.ask(input_filename, stream_size)
        if step == 0: # fill user memory with all 100 users in the first step
            for user_s in stream_user:
                user_in_memory.append(user_s)
                n = 100
        else:
            for user_s in stream_user:
                n += 1
                if random.random() < s/n:
                    pos = random.randint(0, 99)
                    user_in_memory[pos] = user_s
        with open(output_filename, 'a+') as f:
            f.writelines(str(n) + ',' + user_in_memory[0] + ',' + user_in_memory[20] + ',' + user_in_memory[40]
                         + ',' + user_in_memory[60]+ ',' + user_in_memory[80] + "\n")

    print('Duration: ', time.time() - start_time)

