import json

def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

def saveJson(newsJson):
    jsonResult = json.dumps(newsJson)

    # IT_data_full_balanced

    with open("combined_data_IT_with_stock_half_balanced.json", "w") as f:
        f.write(jsonResult)

def mainLoop():
    fileLoc = "json_data/finaldata_half.json"
    newsJson = loadJson(fileLoc)

    hehe = len(newsJson)

    target = list(filter(lambda x: x["companySymbol"].startswith("FB"), newsJson))

    # try to make lambda function to do this job
    count_dec, count_inc = 0, 0
    count_dec = sum(x["label"] == 0 for x in newsJson)
    count_inc = sum(x["label"] == 1 for x in newsJson)

    # obtain label value of data with less quantity
    lessSide = 0 if (count_dec < count_inc) else 1

    # get difference 
    less_dataHolder = [item for item in newsJson if item['label'] == lessSide]

    # add content until count is equal
    newsJson += less_dataHolder[:(count_inc-count_dec)]

    # confirm that the data is balanced
    count_dec2 = sum(x["label"] == 0 for x in newsJson)
    count_inc2 = sum(x["label"] == 1 for x in newsJson)

    if count_dec2 == count_inc2:
        print("Data is balanced.")
        saveJson(newsJson)
    else:
        print("Data imbalanced, code review required.")
    
mainLoop()

print("Data balancing completed.")