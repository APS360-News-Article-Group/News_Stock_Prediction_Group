import requests
import json
import datetime
import holidays
import csv

# define API token
#IEX_API_token = "token=pk_6d20e7d98c5c4e718cefe3c8de1100ec"
IEX_API_token = "token=pk_c6d33eb6c8e74620b88b70caa55965c5"

def validSymbol(stockSymb):
    # define invalid symbol list
    validDict = {}
    for symb in stockSymb:
        try:
            r2 = requests.get(
                f"https://cloud.iexapis.com/stable/stock/{symb}/stats/companyName" + "?" + IEX_API_token)
            validDict[symb] = json.loads(r2.text)
        except:
            continue

    return validDict


def loadCSV(fileLoc):
    # csv obtained here;
    # https://datahub.io/core/s-and-p-500-companies#data

    # using rawList, create an organized JSON dictionary
    symb_toName, name_toSymb, sectorDict = {}, {}, {}
    rawList = []

    with open(fileLoc, mode='r') as f:
        reader = csv.reader(f)

        for row in reader:
            rawList.append(
                {
                    "companySymbol": row[0],
                    "companyName": row[1],
                    "sector": row[2]
                }
            )
            symb_toName[row[0]] = row[1]
            name_toSymb[row[1]] = row[0]

    for item in rawList:
        if item["sector"] not in sectorDict:
            # initialize list
            sectorDict[item["sector"]] = []
        else:
            sectorDict[item["sector"]].append(
                {
                    "companySymbol": item["companySymbol"],
                    "companyName": item["companyName"]
                }
            )

    return rawList, sectorDict, symb_toName, name_toSymb


def loadStockJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data


def nextMarketDate(givenDate):
    next_day = givenDate + datetime.timedelta(days=1)
    while next_day.weekday() in holidays.WEEKEND or next_day in holidays.US():
        next_day += datetime.timedelta(days=1)
    return next_day


def getStock(symb_toName, news_file):
    finalResult, errLog = [], []
    fileLoc_news = news_file
    newsJson = loadStockJson(fileLoc_news)

    for compSymb, compName in symb_toName.items():

        # look into first 3 months
        r1 = requests.get(
            f"https://cloud.iexapis.com/stable/stock/{compSymb}/chart/6m"+"?"+IEX_API_token)

        # disregard corrupted json received from IEX
        try:
            r1_response = json.loads(r1.text)
            r1_dict = {item['date']: item['changePercent']
                       for item in r1_response}

            for item_news in newsJson[compSymb]:
                # check the date published of the news article and return next weekday
                next_MarketOpenDate = nextMarketDate(datetime.datetime.strptime(
                    item_news['date'], '%Y-%m-%d')).strftime('%Y-%m-%d')

                if next_MarketOpenDate in r1_dict:
                    item_news['companyName'] = compName
                    item_news['companySymbol'] = compSymb
                    item_news['changePercent'] = r1_dict[next_MarketOpenDate]
                    item_news['label'] = 1 if (
                        r1_dict[next_MarketOpenDate] > 0) else 0
                    finalResult.append(item_news)

        # use errLog
        except:
            errLog.append((compSymb, compName))

    return finalResult


def mainLoop():
    # target stock NYSE symbol list
    fileLoc = "Tech_industry_list.csv"
    rawList, sectorDict, symb_toName, name_toSymb = loadCSV(fileLoc)

    # valid symb_toName
    symb_toName = validSymbol(tuple(symb_toName.keys()))

    # query IEX Cloud for stock data
    queryResult = getStock(symb_toName, "gnews_data_Jan_Feb.json")
    jsonResult = json.dumps(queryResult)

    with open("gnews_data_Jan_Feb_with_stock.json", "w") as f:
        f.write(jsonResult)

mainLoop()

print("Stock data crawling completed.")
