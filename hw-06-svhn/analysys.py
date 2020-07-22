import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # train   # {1.0: 1753, 2.0: 4718, 3.0: 3322, 4.0: 206, 5.0: 1}
    # dev     # {1.0: 223,  2.0: 576   3.0: 437,  4.0: 29,  5.0: 2}
    # max_numbers = 0

    numbers = {}

    aspects = []
    sizes = []
    heights = []
    anal = []
    with open("10000.txt", 'r') as file:

        for line in file:
            lst = line.split()
            cnt_boxes = ( len(lst) - 1) / 2
            anal.append(cnt_boxes)
            size = int(lst[0])  # height of the image
            sizes.append(size)
            normalizer = size

            for i in range(1, len(lst), 2):
                box_h = int(lst[i])/normalizer
                box_w = int(lst[i+1])/normalizer
                heights.append(box_h)
                aspects.append(box_w/box_h)

            if cnt_boxes in numbers:
                numbers[cnt_boxes] += 1
            else:
                numbers[cnt_boxes] = 1
            #
            # if cnt_boxes > max_numbers:
            #     max_numbers = cnt_boxes
    #
    #
    # print(max_numbers)
    print(numbers)
    aspects = pd.Series(aspects)
    sizes = pd.Series(sizes)
    heights = pd.Series(heights)
    anal = pd.Series(anal)
    print(sizes.describe(percentiles=[0.001, 0.05, 0.1, .25, .5, .75, 0.8, 0.9, 0.95]))
    print("---ASPECT RATIOS----")
    print(aspects.describe(percentiles=[0.001, 0.05, 0.1, .25, .5, .75, 0.9]))
    print("--HEIGHTS---")
    print(heights.describe(percentiles=[0.001, 0.05, 0.1, .25, .5, .75, 0.8, 0.9, 0.95]))
    print(anal.describe(percentiles=[0.001, 0.05, 0.1, .25, .5, 0.6, 0.7, .75, 0.8, 0.9, 0.95]))


    # plt.plot()

    #fig, ax = plt.subplots()

    size = pd.Series(sizes)
    fig, ax = plt.subplots()
    aspects.hist(bins=300, range=(0.2,0.8)).plot(ax=ax, legend=False)

    print(aspects.describe()) # mean 76 ; med 68 ;; 75% 95
    plt.show()



    # resize to 100x1000
    # resize

# --------- TRAIN ------- DEV--------
# IMG SIZE
# mean        76          76
# std         38          40
# min         18          17
# 0.1%        20          21
# 10%         38          38
# 25%         48          48
# 50%         68          67
# 75%         95          95
# 80%        104         103
# 90%        130         132 # ------- resize all images to 130 ???
# max        293         292

# RATIOS [width/height]
# mean         2         2
# std          .5        .5
# min          0         0
# 0.1%         0.87      1.03
# 5%           1.3       1.3
# 10%          1.47      1.5
# 25%          1.6       1
# 50%          1.9       1.9
# 75%          2.3       2.3
# 90%          2.7       2.7
# max          7         5

# HEIGHTS
# mean        42          41
# std         17          17
# min         12          16
# 0.1%        16          16
# 5%          23          22
# 10%         25          25
# 25%         30          30
# 50%         38          37
# 75%         50          49
# 80%         53          53
# 90%         65          64
# 95%         75          74
# max        225         174
