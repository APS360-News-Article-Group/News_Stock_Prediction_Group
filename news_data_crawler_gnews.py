import requests
import json
import csv
import datetime

# gNews for date range = "2019-05-01 ~~ 2019-06-30"
# need the tech company name and stuff

GNews_token = "5d1452bc693a736257a8d00e8ca59ac8"

def loadCSV(fileLoc):
    # csv obtained here;
    # https://datahub.io/core/s-and-p-500-companies#data

    # using rawList, create an organized JSON dictionary
    symb_toName, name_toSymb, sectorDict = {}, {}, {}
    rawList = []

    fileLoc = "C:\\Temp\\S&P500_list.csv"

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


def queryNews(symb_toName):
    # query per day basis, take the top 7 news

    # do it for each company due to api limit
    # loop over each company within the S&P IT industry
    for companySymb, companyName in symb_toName.items():

        GnewsData = {}
        GnewsData[companySymb] = []

        min_date = "2019-05-01"
        max_date = "2019-05-02"
        end_date = "2019-06-30"

        # datetime object conversion
        min_date_dt = datetime.datetime.strptime(min_date, '%Y-%m-%d')
        max_date_dt = datetime.datetime.strptime(max_date, '%Y-%m-%d')
        end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        # need some kind of date range here
        while max_date_dt < end_date_dt:
            
            search_query = companyName
            API_url = f"https://gnews.io/api/v3/search?q={companyName}&mindate={min_date}&maxdate={max_date}"
            r1 = requests.get(API_url + "&token=" + GNews_token)

            r1_response = json.loads(r1.text)

            # do data accumulation
            GnewsData[companySymb].append([{"date": item['date'], 
                                            "headline": item['title'], 
                                            "description": item['description']} 
                                            for item in r1_response])

            min_date_dt += datetime.timedelta(days=1)
            max_date_dt += datetime.timedelta(days=1)
        
        break


def mainLoop():

    fileLoc = "C:\\Temp\\S&P500_list.csv"
    rawList, sectorDict, symb_toName, name_toSymb = loadCSV(fileLoc)

    # specifically interested in sectorDict['Information Technology']

    newsData = queryNews(symb_toName)

    # now start querying the GNews

    print("Yo")


mainLoop()