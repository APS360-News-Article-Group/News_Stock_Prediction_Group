import json

def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

def saveJson(newsJson):
    jsonResult = json.dumps(newsJson)

    with open("C:\\Temp\\finalData_big_balanced.json", "w") as f:
        f.write(jsonResult)


def mainLoop():
    fileLoc = "combined_data_IT_with_stock.json"
    err_list = []

    newsJson = loadJson(fileLoc)

    # Numbers of data in each class
    count_dec, count_inc = 0, 0
    count_dec = sum(x['label'] == 0 for x in newsJson)
    count_inc = sum(x['label'] == 1 for x in newsJson)
    print("Increase:" + str(count_inc))
    print("Increase percentage: " + str(count_inc/ (count_inc + count_dec)))
    print("Decrease:" + str(count_dec))
    print("Decrease percentage: " + str(count_dec/ (count_inc + count_dec)))

    # Date range distribution
    date_distribution = {}
    for x in newsJson:
        date = x['date']
        if date in date_distribution:
            date_distribution[date] += 1
        else:
            date_distribution[date] = 1
    date_range = list(date_distribution.keys())
    date_frequency = list(date_distribution.values())
    print(date_range)
    print(date_frequency)
    # save the date range and frequency information for future use
    with open("date_distribution.csv", "w") as f:
        for i in date_distribution:
            f.write(str(i) + ", " + str(date_distribution[i]) + "\n")

    # Stock range Distribution
    max_change = 0
    min_change = 0
    change_distribution = {
        "-20.0% to -15.01%": 0,
        "-15.0% to -10.01%": 0,
        "-10.0% to -5.01%": 0,
        "-5.0% to -0.01%": 0,
        "0.0% to 4.99%": 0,
        "5.0% to 9.99%": 0,
        "10.0% to 14.99%": 0,
        "15.0% to 19.99%": 0,
        "20.0% to 24.99%": 0
    }
    for x in newsJson:
        change = x['changePercent']
        if change < -15:
            change_distribution["-20.0% to -15.01%"] += 1
        elif change < -10:
            change_distribution["-15.0% to -10.01%"] += 1
        elif change < -5:
            change_distribution["-10.0% to -5.01%"] += 1
        elif change < 0:
            change_distribution["-5.0% to -0.01%"] += 1
        elif change < 5:
            change_distribution["0.0% to 4.99%"] += 1
        elif change < 10:
            change_distribution["5.0% to 9.99%"] += 1
        elif change < 15:
            change_distribution["10.0% to 14.99%"] += 1
        elif change < 20:
            change_distribution["15.0% to 19.99%"] += 1
        else:
            change_distribution["20.0% to 24.99%"] += 1
    print(change_distribution)
    with open("change_distribution.csv", "w") as f:
        for i in change_distribution:
            f.write(str(i) + ", " + str(change_distribution[i]) + "\n")

mainLoop()
