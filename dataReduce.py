import json

# ===== import the py file and it's values =====
import gnews_data_June
newsResult = gnews_data_June.data


def saveJson(newsJson):
    jsonResult = json.dumps(newsJson)

    with open("C:\\Temp\\finalData_big_balanced.json", "w") as f:
        f.write(jsonResult)


def reduceData(topN):
    total_reducedNews = {}

    for key, value in newsResult.items():
        # seen is reset for new company
        reducedNews = []
        seen = {}
        for item in value:
            if item["date"] not in seen:
                reducedNews.append(item)
                seen[item["date"]] = 1
            else:
                if seen[item["date"]] == 5:
                    continue
                else:
                    reducedNews.append(item)
                    seen[item["date"]] += 1
                    
        total_reducedNews[key] = reducedNews

    return total_reducedNews


def mainLoop():
    # need stock data as well wtf... =====
    # wtfffff
    
    total_reducedNews = reduceData(topN = 5)

    # now save...
    
    print("done")

mainLoop()

print("done")