interactions = []
curr = 0
with open("interactions.txt", 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    interaction = []
    for line in lines:
        line = line.split(" ")
        mashup, api = int(line[0]), line[1]
        if mashup == curr:
            interaction.append(api)
        else:
            interactions.append(interaction)
            curr = mashup
            interaction = [api]
    interactions.append(interaction)

# split interactions into train and test sets
train = []
test = []
for interaction in interactions:
    length = len(interaction)
    length = int(length * 0.8)
    train.append(interaction[:length])
    test.append(interaction[length:])

# write train and test sets to files
with open("train.txt", 'w') as f:
    for i, interaction in enumerate(train):
        string = str(i) + " " + " ".join(interaction) + "\n"
        f.write(string)

with open("test.txt", 'w') as f:
    for i, interaction in enumerate(test):
        string = str(i) + " " + " ".join(interaction) + "\n"
        f.write(string)