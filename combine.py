import json
import news_data_template

combined_data = news_data_template.data

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
    with open('json_outputs/newsData_big.json') as json_file:
        bing_data = json.load(json_file)
    with open('json_outputs/newsData_gnews.json') as json_file:
        gnews_data = json.load(json_file)

    # Get company list
    filename = "S&P500_list.csv"
    comp_list = loadCSV(filename)

    # Combine bing and gnews data
    for comp in comp_list:
        comp_result = bing_data[comp] + gnews_data[comp]
        combined_data.update({comp: comp_result})

    # Convert to json
    combined_data_json = json.dumps(combined_data)

    with open('combined_news_data.json','w') as f:
        f.write(combined_data_json)

mainLoop()
