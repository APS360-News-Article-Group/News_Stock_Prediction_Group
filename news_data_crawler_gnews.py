import requests
import json
import pandas as pd
from pandas.tseries.offsets import *
import datetime
import csv
import gnews_data_Jun17_Jun28

# define GNews API token
#token = "b44420c3a8bada2dc8b5f3b5c801d3a5"
#token = "7a988107d7cb0fdd20836cce0499a6ac"
#token = "55846d7fbd99e1a9f2a81dc5c4c65569"
#token = "0a350066ff9fc1fd81c79372ccfda723"  #wame@tempcloud.in
#token = "1e9ef469f25044effc30b1821a67b88e"  #vifipus@armail.in
#token = "a93159b543f2bbee9f393a56e785da8a"  #lwcuyq10852@rmailcloud.com
#token = "1a6d4b8e13069d4df714dcf9e4bdf3de"  #biucqe79620@rmailcloud.com
#token = "0ab95889a3abbe16e9f166108d47df15"  #tnjady96814@rmailcloud.com
#token = "c65787d5af48f8a28e5802f341d5b55b"  #lgejvr63598@rmailcloud.com
token = "dd6fb9cfe67eb88733648ac7c2115af7"  #edhxqp82439@rmailcloud.com
#token = "f1d02583c7cee5d165215393a960996e"  #wame1@tempcloud.in
#token = "5c3e05c169470ef3213ac934d6053b82"  #wame2@tempcloud.in
#token = "28636ba95ff2a2350c2ccc24da9dfd19"  #wame3@tempcloud.in
#token = "335f118f14bd054390ab87c7ceb9558c"  #wame3@tempcloud.in
#token = "03a88c42dbeb142df1feaf5cd40f8d84"  #lwcuyq10852@chacuo.net
#token = "5dcbd54e665a0fd746914a37315cb331"  #biucqe79620@chacuo.net
#token = "ec349b9567c933d53fd358256bea378c"  #tnjady96814@chacuo.net
#token = "dbe495fd8654dd989a3ac6d9be0ba971"  #lgejvr63598@chacuo.net
#token = "8b8405abe7cfcb0cb167cdbd5275565c"  #edhxqp82439@chacuo.net
#token = "88c6b76570b1237363e28716e7b173e2"  #voemhz20975@chacuo.net
#token = "6d20b3e69bd9c2fb0f37060b9354f59f"  #jeixaz05634@chacuo.net
#token = "e8c73cdb4e657ea5aba0138049a85edd"  #acnwjv37986@chacuo.net
#token = "fa05b00920471d2435b8ff880ed2551a"  #lpgxej97561@chacuo.net
#token = "65c18fe1a84a0fd3a60516f785a6f9b6"  #koptsf25483@chacuo.net
#token = "18b3bbdaee855bd278094303d6448311"  #fbkcda95472@chacuo.net
#token = "5c4dc3a90bbe476a23a089c83607db88"  #jrbzni62508@chacuo.net
#token = "919268d87df4f60e9af44b512b98a443"  #pxkiyb40912@chacuo.net
#token = "11db7fce28be8dabcf97187025460f10"  #tcqpri80473@chacuo.net
#token = "7bdc7d3302f37ab4e6e938b28882ce10"  #mauwcn52649@chacuo.net
#token = "4b69b5c56bd30f98a04497c71536a9da"  #briszl64037@chacuo.net
#token = "c0f38456aeb96dbd9c3c013feff56c99"  #ivnwsm62731@chacuo.net
#token = "46c11cce4fdf2989ec74da597e4294c0"  #nkbhjq58327@chacuo.net
#token = "2135a43b480ea33fd5e978f4b3fcb98a"  #zathvq86719@chacuo.net
#token = "4366f298de628c2e258f038ccee058e9"  #yreoza05719@chacuo.net
#token = "20bb59a3863456a701dfd5b4c500e741"  #bfxvmh16795@chacuo.net
#token = "8f3a04d8b4e3befa4e675af1c050a204"  #hqosjy71298@chacuo.net
#token = "f5335d82005ddda0e79fbb7e82da21a4"  #glkbyr39168@chacuo.net
#token = "814594b6458cfb01f60f21aebfcbeaf0"  #glaovy96314@chacuo.net
#token = "9cde2d86ecdc01930312e211d7d5af12"  #zdnmxi68910@chacuo.net
#token = "f0268c8b19983bbbb872d1df201b52f6" #holvtd05348@chacuo.net
#token = "bdbaf05c96e0ed47c4805cfd6e47a89f" #jxzpvm21389@chacuo.net


# import the news data file
newsResult = gnews_data_Jun17_Jun28.data

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


def getNews(symb_toName, comp_start_idx, date_start, business_days_range, target_num):
    queryCount = 0
    target_queryCount = target_num
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

            if queryCount >= target_queryCount:
                print("Starting company and date for next iteration:")
                print("Company index: " + str(comp_idx))
                print("Company name: " + str(compName))
                print("Date: " + str(day))
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
                    company_result = newsResult[compSymb]
                    #print(company_result)
                    company_result.append(query_result)
                    newsResult.update({compSymb: company_result})
                    #print(newsResult)

            queryCount += 1

    return True


def mainLoop():
    csv_filename = "Tech_industry_list.csv"

    # rawList = all company in list
    # sectorDict = organized dictionary of companies by their respective S&P 500 sectors
    rawList, sectorDict, symb_toName, name_toSymb = loadCSV(csv_filename)
    comp_list = []
    for compSymb, compName in symb_toName.items():
        comp_list.append(compName)

    # Get business days in the range March 1 - April 30, 2019
    business_days = getValidDates('2019-06-17', '2019-06-28')
    #print(len(business_days)*len(comp_list))
    #print(business_days)

    # Define where the company and date that we should start, ie. left off from last search
    query_start_date = '2019-06-26'    #default start is '2019-03-01'
    query_start_comp_idx = 70         #default start is 0

    # connect to GNews
    getNews(symb_toName, query_start_comp_idx, query_start_date, business_days, 20)

    with open('gnews_data_Jun17_Jun28.py','w') as f:
        f.write("data = ")
        f.write(str(newsResult))

    # eventually save as a json file: - this will be uncoommented eventually
    #newsResult_json = json.dumps(newsResult)
    #with open("json_outputs/newsData_gnews.json", 'w') as f:
    #    f.write(newsResult_json)
mainLoop()

print("news data crawlering completed.")
