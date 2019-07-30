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
    fileLoc = "C:\\Temp\\IT_news_3.json"
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
        "-25.0% to -6.01%": 0,
        "-6.0% to -5.01%": 0,
        "-5.0% to -4.01%": 0,
        "-4.0% to -3.01%": 0,
        "-3.0% to -2.01%": 0,
        "-2.0% to -1.01%": 0,
        "-1.0% to -0.01%": 0,
        "0.0% to 0.99%": 0,
        "1.0% to 1.99%": 0,
        "2.0% to 2.99%": 0,
        "3.0% to 3.99%": 0,
        "4.0% to 4.99%": 0,
        "5.0% to 5.99%": 0,
        "6.0% to 24.99%": 0
    }
    for x in newsJson:
        change = x['changePercent']
        if change < -6:
            change_distribution["-25.0% to -6.01%"] += 1
        elif change < -5:
            change_distribution["-6.0% to -5.01%"] += 1
        elif change < -4:
            change_distribution["-5.0% to -4.01%"] += 1
        elif change < -3:
            change_distribution["-4.0% to -3.01%"] += 1
        elif change < -2:
            change_distribution["-3.0% to -2.01%"] += 1
        elif change < -1:
            change_distribution["-2.0% to -1.01%"] += 1
        elif change < 0:
            change_distribution["-1.0% to -0.01%"] += 1
        elif change < 1:
            change_distribution["0.0% to 0.99%"] += 1
        elif change < 2:
            change_distribution["1.0% to 1.99%"] += 1
        elif change < 3:
            change_distribution["2.0% to 2.99%"] += 1
        elif change < 4:
            change_distribution["3.0% to 3.99%"] += 1
        elif change < 5:
            change_distribution["4.0% to 4.99%"] += 1
        elif change < 6:
            change_distribution["5.0% to 5.99%"] += 1
        else:
            change_distribution["6.0% to 24.99%"] += 1

    print(change_distribution)
    with open("change_distribution.csv", "w") as f:
        for i in change_distribution:
            f.write(str(i) + ", " + str(change_distribution[i]) + "\n")

mainLoop()
