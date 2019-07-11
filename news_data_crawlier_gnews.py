import requests
import json
import pandas as pd
from pandas.tseries.offsets import *
import datetime
import csv
import gnews_data

# define GNews API token
token = "7a988107d7cb0fdd20836cce0499a6ac"

# import the news data file
newsResult = gnews_data.data

def loadCSV(filename):
    # csv obtained here;
    # https://datahub.io/core/s-and-p-500-companies#data
    symb_toName = {}
    name_toSymb = {}
    rawList = []
    with open(filename, mode='r') as f:
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

# Get business days in the range
def getValidDates(start, end):
    valid_dates = pd.bdate_range(start, end)
    valid_dates_list = valid_dates.strftime("%Y-%m-%d")
    return valid_dates_list


def queryStringBuilder(companyName, date):
    target_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    start_date = target_date + datetime.timedelta(days=-1)
    end_date = target_date + datetime.timedelta(days=+1)

    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    url = f"https://gnews.io/api/v3/search?q={companyName}&mindate={from_date}&maxdate={to_date}&token={token}"
    return url


# function used for debugging - not actually used
def blah(comp, date):
    url = queryStringBuilder(comp, date)
    #r1 = requests.get(url).json()['articles'][0]['title']
    r1 = requests.get(url).json()
    cnt = r1['articleCount']
    art = r1['articles']
    print(cnt)
    print(art)
    return r1


def getNews(symb_toName, comp_start_idx, date_start, business_days_range):
    queryCount = 0
    target_queryCount = 10
    # Max 10 items per query instance
    perquery_dataCount = 10
    comp_idx = -1
    findstart = True

    for compSymb, compName in symb_toName.items():
        comp_idx += 1
        if comp_idx < comp_start_idx:
            print("skipped company:" + str(comp_idx))
            continue

        news_result = []
        for day in business_days_range:

            # If this is the first company, we don't start from the very first day;
            # Rather, we find the date where we left off last query
            if findstart:
                if day != date_start:
                    print("skipped date:" + str(day))
                    continue
                else:
                    findstart = False

            if queryCount > target_queryCount:
                print("Starting company and date for next iteration")
                print(comp_idx)
                print(compName)
                print(day)
                return True

            url = queryStringBuilder(compName, day)
            #print(url)
            print(compName)
            print(day)

            r1 = requests.get(url).json()
            #print("r1")
            #print(r1)
            #print("\n")

            if r1['articleCount'] > 0:
                for article in r1['articles']:
                    query_result = {"date": day, "title": article['title'], "description": article["description"]}
                    company_result = newsResult[compName]
                    #print(company_result)
                    company_result.append(query_result)
                    newsResult.update({compName: company_result})
                    #print(newsResult)

            queryCount += 1

    return True


def mainLoop():
    filename = "S&P500_list.csv"

    # rawList = all company in list
    # sectorDict = organized dictionary of companies by their respective S&P 500 sectors
    rawList, sectorDict, symb_toName, name_toSymb = loadCSV(filename)
    comp_list = []
    for compSymb, compName in symb_toName.items():
        comp_list.append(compName)

    # Get business days in the range March 1 - April 30, 2019
    business_days = getValidDates('2019-03-01', '2019-04-30')
    #print(business_days)

    # Define where the company and date that we should start, ie. left off from last search
    query_start_date = '2019-04-18'
    query_start_comp_idx = 1

    # connect to GNews
    getNews(symb_toName, query_start_comp_idx, query_start_date, business_days)

    with open('gnews_data.py','w') as f:
        f.write("data = ")
        f.write(str(newsResult))

mainLoop()

print("news data crawlering completed.")
