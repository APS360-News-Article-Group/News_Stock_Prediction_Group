import json

with open('bing_news_data_IT_with_stock.json') as json_file:
    bing_data = json.load(json_file)
with open('gnews_data_Mar_Apr_IT_with_stock.json') as json_file:
    gnews_data = json.load(json_file)

combined_data = bing_data
combined_data.append(gnews_data)
json_data = json.dumps(combined_data)

with open('combined_data_IT_with_stock.json', 'w') as f:
    f.write(json_data)
