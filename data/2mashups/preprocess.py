import json

with open("../api_name.json", "r") as f:
    api_names = json.load(f)

with open("../api_category.json", "r") as f:
    api_categories = json.load(f)

with open("../mashup_name.json", "r") as f:
    mashup_names = json.load(f)

with open("../mashup_category.json", "r") as f:
    mashup_categories = json.load(f)

with open("../mashup_used_api.json", "r") as f:
    mashup_used_apis = json.load(f)

warm_start_mashup_indices = []

for i, mashup_used_api in enumerate(mashup_used_apis):
    if len(mashup_used_api) >= 2:
        warm_start_mashup_indices.append(i)

warm_start_api_indices = []
warm_start_mashup_api_interactions = []
for mashup_i, warm_start_mashup_index in enumerate(warm_start_mashup_indices):
    interactions = []
    mashup_used_api = mashup_used_apis[warm_start_mashup_index]
    # print(mashup_name)
    for api_name in mashup_used_api:
        api_index = api_names.index(api_name)
        if api_index not in warm_start_api_indices:
            warm_start_api_indices.append(api_index)
        interactions.append(warm_start_api_indices.index(api_index))
    warm_start_mashup_api_interactions.append(interactions)

print("Warm start mashup indices:", len(warm_start_mashup_indices))
print("Warm start API indices:", len(warm_start_api_indices))

print("Warm start mashup API interactions:", warm_start_mashup_api_interactions[:5])

with open("train.txt", "w") as f:
    with open("test.txt", "w") as f_test:
        for i, interactions in enumerate(warm_start_mashup_api_interactions):
            train_size = int(len(interactions) * 0.8)
            train_interactions = interactions[:train_size]
            test_interactions = interactions[train_size:]
            train_string = " ".join([str(x) for x in train_interactions])
            test_string = " ".join([str(x) for x in test_interactions])
            train_string = str(i) + " " + train_string
            test_string = str(i) + " " + test_string
            f.write(train_string + "\n")
            f_test.write(test_string + "\n")
f_test.close()
f.close()

# build co_category-api matrix
co_category_api_matrix = []
for i, warm_start_api_index in enumerate(warm_start_api_indices):
    i_categories = api_categories[warm_start_api_index]
    for j, warm_start_api_index in enumerate(warm_start_api_indices):
        if i == j:
            continue
        j_categories = api_categories[warm_start_api_index]
        common_categories = set(i_categories) & set(j_categories)
        if len(common_categories) > 0:
            co_category_api_matrix.append((i, j, len(common_categories)))

with open("api_co_category.txt", "w") as f:
    for i, j, count in co_category_api_matrix:
        f.write(f"{i} {j} {count}\n")
f.close()

co_category_mashup_matrix = []
for i, warm_start_mashup_index in enumerate(warm_start_mashup_indices):
    i_categories = mashup_categories[warm_start_mashup_index]
    for j, warm_start_mashup_index in enumerate(warm_start_mashup_indices):
        if i == j:
            continue
        j_categories = mashup_categories[warm_start_mashup_index]
        common_categories = set(i_categories) & set(j_categories)
        if len(common_categories) > 0:
            co_category_mashup_matrix.append((i, j, len(common_categories)))

with open("mashup_co_category.txt", "w") as f:
    for i, j, count in co_category_mashup_matrix:
        f.write(f"{i} {j} {count}\n")
f.close()

print("Co-category API matrix:", len(co_category_api_matrix))
print("Co-category Mashup matrix:", len(co_category_mashup_matrix))

co_api_mashup_matrix = []
for i, warm_start_mashup_api_interaction in enumerate(warm_start_mashup_api_interactions):
    i_interactions = warm_start_mashup_api_interaction
    for j, warm_start_api_index in enumerate(warm_start_api_indices):
        if j in i_interactions:
            continue
        j_interactions = warm_start_mashup_api_interactions[j]
        common_interactions = set(i_interactions) & set(j_interactions)
        if len(common_interactions) > 0:
            co_api_mashup_matrix.append((i, j, len(common_interactions)))

with open("mashup_co_api.txt", "w") as f:
    for i, j, count in co_api_mashup_matrix:
        f.write(f"{i} {j} {count}\n")
f.close()

print("Co-api mashup matrix:", len(co_api_mashup_matrix))

import numpy as np

co_mashup_api_matrix = np.zeros((len(warm_start_api_indices), len(warm_start_api_indices)))
for interactions in warm_start_mashup_api_interactions:
    for i in interactions:
        for j in interactions:
            if i != j:
                co_mashup_api_matrix[i][j] += 1

# save as numpy file
np.save("co_mashup_api_matrix.npy", co_mashup_api_matrix)
# save as csr matrix
from scipy.sparse import csr_matrix, save_npz
co_mashup_api_csr = csr_matrix(co_mashup_api_matrix)
with open("co_mashup_api_matrix.npz", "wb") as f:
    save_npz(f, co_mashup_api_csr)