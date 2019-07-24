import json

def loadStockJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

def dataSplit():
    fileLoc = "C:\\Temp\\IT_news_2_balanced.json"
    trainLoc = "C:\\Temp\\IT_train.json"
    validLoc = "C:\\Temp\\IT_valid.json"
    testLoc = "C:\\Temp\\IT_test.json"
    posList, negList, trainList, validList, testList = [], [], [], [], []
    switchData = True
    a, b = 0, 0

    dataDict = loadStockJson(fileLoc)
    trainCount = int(len(dataDict) * 0.7)

    # separate items based on their labels for 50 : 50 train, valid, test dataset distribution
    for item in dataDict:
        if item["label"] == 1:
            posList.append(item)
        else:
            negList.append(item)
    
    # now order posList and negList by their dates
    posList = sorted(posList, key=lambda x: x["date"])
    negList = sorted(negList, key=lambda x: x["date"])

    trainList = posList[:int(trainCount/2)] + negList[:trainCount-int(trainCount/2)]

    posList = posList[int(trainCount/2):]
    negList = negList[trainCount-int(trainCount/2):]

    for i, item in enumerate(posList):
        if switchData:
            validList.append(item)
            switchData = False
        else:
            testList.append(item)
            switchData = True

    for i, item in enumerate(negList):
        if switchData:
            validList.append(item)
            switchData = False
        else:
            testList.append(item)
            switchData = True

    trainResult = json.dumps(trainList)
    validResult = json.dumps(validList)
    testResult = json.dumps(testList)

    with open(trainLoc, "w") as f:
        f.write(trainResult)

    with open(validLoc, "w") as f:
        f.write(validResult)

    with open(testLoc, "w") as f:
        f.write(testResult)

dataSplit()

print("Data split finished.")
