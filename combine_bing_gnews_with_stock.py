import json

with open('combined_data_IT_with_stock_May_June.json') as json_file:
    data1 = json.load(json_file)
with open('combined_data_IT_with_stock_Mar_Apr_bing.json') as json_file:
    data2 = json.load(json_file)
with open('gnews_data_Jan_Feb_with_stock.json') as json_file:
    data3 = json.load(json_file)
#print("bing: " + str(len(bing_data)))
#print("gnews: " + str(len(gnews_data)))
combined_data = data1 + data2 + data3
print("total: " + str(len(combined_data)))
json_data = json.dumps(combined_data)

with open('combined_data_IT_with_stock_full.json', 'w') as f:
    f.write(json_data)
