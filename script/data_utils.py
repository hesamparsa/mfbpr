import urllib.request
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# download-unzip and/or load the data
def read_data_ml100k(data_path, delimiter, col_names, url):
    """load movie lens data."""
    if np.logical_not(os.path.isdir(data_path)):
        zip_path, _ = urllib.request.urlretrieve(url)
        with zipfile.ZipFile(zip_path, "r") as f:
            print(pd.Series(f.namelist()))
            f.extractall()
    data = pd.read_csv(data_path + 'u.data', delimiter, names=col_names)
    num_users, num_items = (data.user_id.nunique(), data.item_id.nunique())
    sparsity = 1 - len(data) / (num_users * num_items)
    print(f'number of users: {num_users}, number of items: {num_items}')
    print(f'matrix sparsity: {sparsity:f}')
    return data, num_users, num_items, sparsity

# Split into test-train
def split_data(data, num_users, num_items, split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [
            True if x == 1 else False
            for x in np.random.uniform(0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data

# create interaction file
def load_data(data, num_users, num_items, feedback='explicit'):
    """Transform data into appropriate format."""
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter