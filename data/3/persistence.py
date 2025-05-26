import numpy as np

interactions = []
with open("./api_co_category.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" ", 1)
        api1, api2 = line[0], line[1]
        if {api1, api2} not in interactions and {api2, api1} not in interactions:
            interactions.append({api1, api2})
        else:
            continue

print(len(interactions))

# 从interactions中随机sample5000个
import random
sampled_interactions = interactions
api_count = np.zeros(1837)
for i, interaction in enumerate(sampled_interactions):
    api1, api2 = list(interaction)
    api1, api2 = int(api1), int(api2)
    api_count[api1] += 1
    api_count[api2] += 1

zero_indices = np.where(api_count == 0)[0]
print(zero_indices)

for i in zero_indices:
    for j in range(len(interactions)):
        if str(i) in interactions[j]:
            sampled_interactions.append(interactions[j])
            break

print(len(sampled_interactions))

# with open("simplified_mashup_co_category.txt", "w") as f:
#     for interaction in sampled_interactions:
#         api1, api2 = list(interaction)
#         f.write(f"{api1} {api2}\n")

with open("api_co_category.txt", "w") as f:
    for interaction in sampled_interactions:
        api1, api2 = list(interaction)
        f.write(f"{api1} {api2}\n")


