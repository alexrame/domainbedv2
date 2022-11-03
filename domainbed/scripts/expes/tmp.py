num_weightings = 5
half_num_weightings = (num_weightings - 1) // 2
weightings = [0.5 + 0.5 * (c-half_num_weightings) / max(1, half_num_weightings) for c in range(num_weightings)]

