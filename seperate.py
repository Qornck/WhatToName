train_file = './data/3/train.txt'
trainUser = []
trainItem = []

with open(train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            trainUser.extend([uid] * len(items))
            trainItem.extend(items)

test_file = './data/3/test.txt'
testUser = []
testItem = []

with open(test_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            try:
                items = [int(i) for i in l[1:]]
            except:
                print(l[0])
            uid = int(l[0])
            testUser.extend([uid] * len(items))
            testItem.extend(items)

with open('./train.txt', 'w') as f:
    for i in range(len(trainUser)):
        f.write(str(trainUser[i]) + ',' + str(trainItem[i]) + '\n')

with open('./test.txt', 'w') as f:
    for i in range(len(testUser)):
        f.write(str(testUser[i]) + ',' + str(testItem[i]) + '\n')