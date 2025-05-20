import numpy as np
train_file = './data/3/train.txt'
trainUser = []
trainItem = []
train = []

with open(train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            for item in items:
                train.append(np.array([uid, item]))

train = np.array(train)

test_file = './data/3/test.txt'
testUser = []
testItem = []
test = []

with open(test_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            try:
                items = [int(i) for i in l[1:]]
            except:
                print(l[0])
            uid = int(l[0])
            for item in items:
                test.append(np.array([uid, item]))

test = np.array(test)

np.save("train_list.npy", train)
np.save("test_list.npy", test)

# with open('./train.txt', 'w') as f:
#     for i in range(len(trainUser)):
#         f.write(str(trainUser[i]) + ',' + str(trainItem[i]) + '\n')

# with open('./test.txt', 'w') as f:
#     for i in range(len(testUser)):
#         f.write(str(testUser[i]) + ',' + str(testItem[i]) + '\n')