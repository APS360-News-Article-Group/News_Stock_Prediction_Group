import requests
import json
import datetime
import csv
from azure.cognitiveservices.search.newssearch import NewsSearchAPI
from msrest.authentication import CognitiveServicesCredentials

# define bing API subscription key
subscription_key = "9a5a1c9b93184f71ac35f2237ef508ca"
targetComp = "ACN ATVI ADBE AMD AKAM ADS GOOGL GOOG APH ADI ANSS AAPL AMAT ADSK ADP AVGO CA CDNS CSCO CTXS CTSH GLW CSRA DXC EBAY EA FFIV FB FIS FISV FLIR IT GPN HRS HPE HPQ INTC IBM INTU IPGP JNPR KLAC LRCX MA MCHP MU MSFT MSI NTAP NFLX NVDA ORCL PAYX PYPL QRVO QCOM RHT CRM STX SWKS SYMC SNPS TTWO TEL TXN TSS VRSN V WDC WU XRX XLNX"
targetCompList = targetComp.split(" ")


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

    try:
        for compSymb, compName in symb_toName.items():
            client = NewsSearchAPI(CognitiveServicesCredentials(subscription_key))

            mySet = set()
            perquery_dataCount = 80
            news_result = []

            news_result = client.news.search(
                    query=compName,
                    count=perquery_dataCount,
                    market="en-us",
                    sort_by="Date").value

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
    except:
        return newsData

    return newsData


def filterIT(symb_toName):
    symb_toName_IT = {}
    name_toSymb_IT = {}
    for key, value in symb_toName.items():
        if key in targetCompList:
            symb_toName_IT[key] = value
            name_toSymb_IT[value] = key
    
    return symb_toName_IT, name_toSymb_IT


def mainLoop():
    newsResult = []

    fileLoc = "C:\\Temp\\S&P500_list.csv"

    # rawList = all company in list
    # sectorDict = organized dictionary of companies by their respective S&P 500 sectors
    rawList, sectorDict, symb_toName, name_toSymb = loadCSV(fileLoc)

    symb_toName_IT, name_toSymb_IT = filterIT(symb_toName)

    # for data cut off
    new_dict = {}
    i = 0
    for key, value in symb_toName_IT.items():
        i += 1
        if i >= 41:
            new_dict[key] = value

    # connect to bing
    newsData = getNews(new_dict)

    jsonResult = json.dumps(newsData)

    with open("C:\\Temp\\newsData_bing_IT.json", "w") as f:
        f.write(jsonResult)


mainLoop()

print("news data crawlering completed.")
