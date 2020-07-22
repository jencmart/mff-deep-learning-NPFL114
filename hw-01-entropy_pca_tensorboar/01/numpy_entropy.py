#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # TODO: Create a NumPy array containing the model distribution.
    # Load model distribution, each line `string \t probability`.
    model_data = {}
    with open("01/numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            x = line.split("\t")
            model_data[x[0]] = x[1]

    # Load data distribution, each line containing a datapoint -- a string.
    data_cnt = {}
    for key in model_data:
        data_cnt[key] = 0

    cnt = 0
    is_inf = False
    with open("01/numpy_entropy_data.txt", "r") as data_prob:
        for line in data_prob:
            cnt += 1
            line = line.rstrip("\n")
            if line in data_cnt:
                data_cnt[line] += 1
            else:
                data_cnt[line] = 1
                is_inf = True
                model_data[line] = 0.0

    x = []
    for key in sorted(model_data.keys()):
        x.append(model_data[key])
    model_prob = np.array(x, dtype=np.float)

    x = []
    for key in sorted(data_cnt.keys()):
        x.append(data_cnt[key])
    data_prob = np.array(x, dtype=np.float)

    data_prob = data_prob / cnt  # empiric probabilities



    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    mult = np.nan_to_num(np.multiply(data_prob, np.log(data_prob)))
    entropy = -1 * mult.sum()
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # print(data)
    # print(model_data)
    if is_inf:
        print(np.inf)
        print(np.inf)
    else:
        mult = np.nan_to_num(np.multiply(data_prob, np.log(model_prob)))
        entropy = -1 * mult.sum()
        print("{:.2f}".format(entropy))

        # and KL-divergence D_KL(data distribution, model_distribution)
        mult = np.nan_to_num(np.multiply(data_prob, np.log(data_prob / model_prob)))
        entropy = mult.sum()
        print("{:.2f}".format(entropy))
