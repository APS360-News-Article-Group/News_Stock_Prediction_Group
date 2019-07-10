import requests
import json
import datetime
import csv
from azure.cognitiveservices.search.newssearch import NewsSearchAPI
from msrest.authentication import CognitiveServicesCredentials

# define bing API subscription key
subscription_key = "4d14b5ff3d5f43819027b76f89809472"


def loadCSV(fileLoc):

    # csv obtained here;
    # https://datahub.io/core/s-and-p-500-companies#data
    symb_toName = {}
    name_toSymb = {}
    rawList = []
    fileLoc = "C:\\Temp\\S&P500_list.csv"
    with open(fileLoc, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            rawList.append(
                {"companySymbol": row[0], "companyName": row[1], "sector": row[2]})
            symb_toName[row[0]] = row[1]
            name_toSymb[row[1]] = row[0]

    # using rawList, create an organized JSON dictionary
    sectorDict = {}

    for item in rawList:
        if item["sector"] not in sectorDict:
            # initialize list
            sectorDict[item["sector"]] = []
        else:
            sectorDict[item["sector"]].append(
                {"companySymbol": item["companySymbol"], "companyName": item["companyName"]})

    return rawList, sectorDict, symb_toName, name_toSymb


def loadStockJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data


def getNews(symb_toName):
    newsData = {}

    for compSymb, compName in symb_toName.items():
        client = NewsSearchAPI(CognitiveServicesCredentials(subscription_key))

        mySet = set()
        mySet2 = set()

        # Max 40 items per query instance
        # 40+2 is due to the way data is indexed. This returns 40 data
        target_dataCount = 80
        perquery_dataCount = 40
        news_result = []

        for i in range(0, int(target_dataCount/perquery_dataCount)):
            news_result += client.news.search(
                query=compName,
                count=perquery_dataCount + 2,
                offset=str(0+perquery_dataCount*i),
                market="en-us").value

        # receive a unique url list for duplicate avoidance
        for item in news_result:
            mySet.add(item.name)

        myDict = dict.fromkeys(mySet, 'unseen')

        # mySet now contains unique URLs
        newsData[compSymb] = []

        for item in news_result:
            if item.name in myDict and myDict[item.name] == 'unseen':

                newsData[compSymb].append(
                    {"date": item.date_published[:10],
                     "headline": item.name,
                     "description": item.description,
                     "url": item.url})

                myDict[item.name] = 'seen'

    return newsData


def mainLoop():
    newsResult = []

    fileLoc = "C:\\Temp\\S&P500_list.csv"

    # rawList = all company in list
    # sectorDict = organized dictionary of companies by their respective S&P 500 sectors
    rawList, sectorDict, symb_toName, name_toSymb = loadCSV(fileLoc)

    # connect to bing
    newsData = getNews(symb_toName)

    jsonResult = json.dumps(newsData)

    with open("C:\\Temp\\newsData_big.json", "w") as f:
        f.write(jsonResult)


mainLoop()

print("news data crawlering completed.")
