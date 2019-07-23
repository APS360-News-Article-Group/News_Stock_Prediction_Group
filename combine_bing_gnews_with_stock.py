import json

with open('bing_news_data_IT_with_stock.json') as json_file:
    bing_data = json.load(json_file)
with open('gnews_data_Mar_Apr_IT_with_stock.json') as json_file:
    gnews_data = json.load(json_file)

print("bing: " + str(len(bing_data)))
print("gnews: " + str(len(gnews_data)))
combined_data = bing_data + gnews_data
print("total: " + str(len(combined_data)))
json_data = json.dumps(combined_data)

with open('combined_data_IT_with_stock.json', 'w') as f:
    f.write(json_data)
