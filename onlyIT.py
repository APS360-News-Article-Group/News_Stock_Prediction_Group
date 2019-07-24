import json

def loadStockJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

def mainLoop():
    fileLoc = "C:\\Temp\\finalData_big.json"
    targetComp = "ACN ATVI ADBE AMD AKAM ADS GOOGL GOOG APH ADI ANSS AAPL AMAT ADSK ADP AVGO CA CDNS CSCO CTXS CTSH GLW CSRA DXC EBAY EA FFIV FB FIS FISV FLIR IT GPN HRS HPE HPQ INTC IBM INTU IPGP JNPR KLAC LRCX MA MCHP MU MSFT MSI NTAP NFLX NVDA ORCL PAYX PYPL QRVO QCOM RHT CRM STX SWKS SYMC SNPS TTWO TEL TXN TSS VRSN V WDC WU XRX XLNX"
    targetCompList = targetComp.split(" ")
    
    dataDict = loadStockJson(fileLoc)

    ITlist = []

    for item in dataDict:
        if item["companySymbol"] in targetCompList:
            ITlist.append(item)

    jsonResult = json.dumps(ITlist)

    with open("C:\\Temp\\IT_news.json", "w") as f:
        f.write(jsonResult)

def readCount():
    a, b = 0, 0
    fileLoc = "C:\\Temp\\IT_news.json"
    dataDict = loadStockJson(fileLoc)

    for item in dataDict:
        if item["label"] == 1:
            a += 1
        else:
            b += 1
    print(a, b)
    print(a/len(dataDict), b/len(dataDict))
    print(len(dataDict))

readCount()
# mainLoop()