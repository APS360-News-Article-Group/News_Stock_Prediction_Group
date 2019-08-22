import json


def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data


def saveJson(newsJson):
    jsonResult = json.dumps(newsJson)

    # IT_data_full_balanced

    with open("C:\\Temp\\finalData_big_balanced_keyTitle.json", "w") as f:
        f.write(jsonResult)


def mainLoop():
    fileLoc = "C:\\Temp\\finalData_big_balanced_fix.json"
    newsJson = loadJson(fileLoc)

    i = len(newsJson)

    newData = []

    for item in newsJson:
        if "title" not in item:
            item["title"] = item["headline"]
            del item["headline"]

        newData.append(item)
    
    # now replace the old file
    saveJson(newData)
    
mainLoop()

print("done")