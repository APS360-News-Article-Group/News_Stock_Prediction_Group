import json

def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data


def saveJson(newsJson):
    jsonResult = json.dumps(newsJson)

    with open("combined_data_IT_with_stock_half.json", "w") as f:
        f.write(jsonResult)


def reduceData(topN, newsJson):
    total_reducedNews = {}
    seen = {}
    num_items = 0
    reducedNews = []

    for item in newsJson:
        if item["companySymbol"] not in seen:
            seen[item["companySymbol"]] = {}

        if item["date"] not in seen[item["companySymbol"]]:
            seen[item["companySymbol"]][item["date"]] = 0
        else:
            if seen[item["companySymbol"]][item["date"]] == 5:
                continue
            else:
                reducedNews.append(item)
                seen[item["companySymbol"]][item["date"]] += 1

    return reducedNews


def mainLoop():
    fileLoc = "combined_data_IT_with_stock_full.json"
    newsJson = loadJson(fileLoc)
    
    reducedNews = reduceData(topN = 5, newsJson=newsJson)

    saveJson(reducedNews)
    

mainLoop()

print("done")