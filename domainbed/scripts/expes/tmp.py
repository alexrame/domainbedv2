num_weightings = 5
half_num_weightings = (num_weightings - 1) // 2
weightings = [0.5 + 0.5 * (c-half_num_weightings) / max(1, half_num_weightings) for c in range(num_weightings)]

import math
def filter_out(i_weighting, rank_checkpoint, half_len_checkpoint):
    max_len = i_weighting * half_len_checkpoint
    assert math.abs(int(max_len) - max_len) < 0.000001
    max_len = round(max_len)
    if rank_checkpoint < max_len:
        return 1
    else:
        return 0
ranks = range(10)
half_len_checkpoint = 10
for rank_checkpoint in ranks:
    for i_w in range(0, 1, 11):
        print(rank_checkpoint, i_w)
        print(filter_out(i_w, rank_checkpoint, half_len_checkpoint))
