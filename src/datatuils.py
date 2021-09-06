import os


def read_train_file():
    with open('../data/toy_train.txt', 'r', encoding='utf8') as f:
        i = 0
        for line in f.readlines():
            print(line)
            strs = line.split('\t')
            for str in strs:
                print(str)
            print("--------------------")
            i = i + 1
            if i == 10:
                break


read_train_file()
