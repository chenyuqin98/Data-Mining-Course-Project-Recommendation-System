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
function_num = 16
hash_para_a = random.sample(range(1, sys.maxsize - 1), function_num)
hash_para_b = random.sample(range(0, sys.maxsize - 1), function_num)

array_length = 69997
A = [0]*array_length
prev_user_set = set() # used for fpr



def myhashs(s):
    function_rlt = []
    str_int_val = int(binascii.hexlify(s.encode('utf8')), 16)
    for i in range(function_num):
        hash_val = (hash_para_a[i]*str_int_val+hash_para_b[i]) % array_length
        function_rlt.append(hash_val)
    return function_rlt

def check(s):
    """return 1 if new object is in S"""
    s_hash_rlt = myhashs(s)
    for hash_val in s_hash_rlt:
        if A[hash_val] == 0:
            return 0
    return 1

def construct(s):
    s_hash_rlt = myhashs(s)
    for hash_val in s_hash_rlt:
        A[hash_val] = 1

def calculate_fpr(fp, tn, step):
    # print(step, fp, tn)
    fpr = fp / (fp+tn)
    with open(output_filename, 'a+') as f:
        f.writelines(str(step)+','+str(fpr)+"\n")

if __name__=='__main__':
    start_time = time.time()
    bx = BlackBox()

    with open(output_filename, 'w+') as f:
        f.writelines("Time,FPR\n")

    for step in range(num_of_asks):
        stream_user = bx.ask(input_filename, stream_size)
        predict = []
        tn = 0
        fp = 0
        for user_s in stream_user:
            predict.append(check(user_s))
            construct(user_s)
            if user_s not in prev_user_set:
                tn += 1
            if predict[-1] == 1 and user_s not in prev_user_set:
                fp += 1
            prev_user_set.add(user_s)
        calculate_fpr(fp, tn, step)

    print('Duration: ', time.time() - start_time)

