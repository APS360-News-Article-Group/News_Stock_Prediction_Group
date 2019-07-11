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
    compList = []
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            compList.append(row[0])
    return compList

def mainLoop():
    newsResult = []

    filename = "S&P500_list.csv"

    # rawList = all company in list
    # sectorDict = organized dictionary of companies by their respective S&P 500 sectors
    comp_list = loadCSV(filename)
    data_template = {comp: [] for comp in comp_list}
    print(data_template)

    with open('gnews_data.py','w') as f:
        f.write("data = ")
        f.write(str(data_template))



mainLoop()

print("news data crawlering completed.")
