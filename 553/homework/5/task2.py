from blackbox import BlackBox
import random
import time
import sys
import binascii

input_filename = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_filename = sys.argv[4]

# create hash functions
function_num = 10
hash_para_a = random.sample(range(1, sys.maxsize - 1), function_num)
hash_para_b = random.sample(range(0, sys.maxsize - 1), function_num)

array_length = 69997
A = [0]*array_length
prev_user_set = set() # used for fpr

# for test
estimate_total, ground_truth_total = 0, 0



def myhashs(s):
    function_rlt = []
    str_int_val = int(binascii.hexlify(s.encode('utf8')), 16)
    for i in range(function_num):
        hash_val = (hash_para_a[i]*str_int_val+hash_para_b[i]) % array_length
        function_rlt.append(hash_val)
    return function_rlt

def estimate_and_count(stream_user, step):
    global estimate_total, ground_truth_total
    user_set = set()  # ground truth
    zero_num = [0] * function_num
    for user_s in stream_user:
        user_set.add(user_s)
        user_hash_vals = myhashs(user_s)
        for i in range(function_num):
            v = user_hash_vals[i]
            zero_num[i] = max(zero_num[i], len(bin(v).split('1')[-1]))
    estimations = [2**r for r in zero_num]
    # estimate = sum(estimations) / len(estimations)
    # estimations.sort()
    # estimate = int(estimations[int(len(estimations) / 2)])
    group_length = 1
    group_number = int(function_num/group_length)
    group_avg = []
    for i in range(group_number):
        avg = sum(estimations[i:i+group_length])/group_length
        group_avg.append(avg)
    group_avg.sort()
    estimate = int(group_avg[int(len(group_avg) / 2)])

    ground_truth = len(user_set)
    estimate_total += estimate
    ground_truth_total += ground_truth
    with open(output_filename, 'a+') as f:
        f.writelines(str(step)+','+str(ground_truth)+','+str(estimate)+"\n")

if __name__=='__main__':
    start_time = time.time()
    bx = BlackBox()

    with open(output_filename, 'w+') as f:
        f.writelines("Time,Ground Truth,Estimation\n")

    for step in range(num_of_asks):
        stream_user = bx.ask(input_filename, stream_size)
        estimate_and_count(stream_user, step)

    print(ground_truth_total, estimate_total, estimate_total/ground_truth_total)
    print('Duration: ', time.time() - start_time)

