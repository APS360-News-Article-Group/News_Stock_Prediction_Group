import requests
import json
import pandas as pd
from pandas.tseries.offsets import *
import datetime
import csv

# define GNews API token
token = "55846d7fbd99e1a9f2a81dc5c4c65569"


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

def mainLoop():
    newsResult = []

    filename = "S&P500_list.csv"

    # rawList = all company in list
    # sectorDict = organized dictionary of companies by their respective S&P 500 sectors
    rawList, sectorDict, symb_toName, name_toSymb = loadCSV(filename)
    comp_list = []
    for compSymb, compName in symb_toName.items():
        comp_list.append(compName)
    data_template = {comp: [] for comp in comp_list}
    print(data_template)

    with open('gnews_data.py','w') as f:
        f.write("data = ")
        f.write(str(data_template))




mainLoop()

print("news data crawlering completed.")
