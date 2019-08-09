import json


def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data


def mainLoop():
    # choose depending on your training dataset
    fileLoc = "finaldata_full.json"
    fileLoc = "finaldata_half.json"

    newsJson = loadJson(fileLoc)

    # first split to two list, labelled 1 and labelled 0
    label_1, label_0 = [], []

    total_len = len(newsJson)

    for item in newsJson:
        if item["label"] == 1:
            label_1.append(item)
        else:
            label_0.append(item)

    # now order them based on dates
    label_1_sorted = sorted(label_1, key=lambda x: x["date"])
    label_0_sorted = sorted(label_0, key=lambda x: x["date"])

    # distribute 70% to the training, 15% each with randomized indexing for validation / testing
    train_list, valid_list, test_list = [], [], []

    train_list += label_1_sorted[:int(total_len * 0.35)]
    train_list += label_0_sorted[:int(total_len * 0.35)]

    label_1_sorted = label_1_sorted[int(total_len * 0.35):]
    label_0_sorted = label_0_sorted[int(total_len * 0.35):]

    for i, (item_1, item_0) in enumerate(zip(label_1_sorted, label_0_sorted)):
        if i % 2 == 0:
            valid_list.append(item_1)
            valid_list.append(item_0)
        else:
            test_list.append(item_1)
            test_list.append(item_0)

    # final check
    total_len_list = len(train_list) + len(valid_list) + len(test_list)

    if total_len_list == total_len:
        print("length check sucessful")
        return train_list, valid_list, test_list
    else:
        print("length check failed. Please review the code.")
        return None


train_list, valid_list, test_list = mainLoop()

print("done")
