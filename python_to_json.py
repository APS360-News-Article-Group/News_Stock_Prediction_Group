import json
import gnews_data_Jan_Feb

data = gnews_data_Jan_Feb.data

jsonResult = json.dumps(data)

with open("gnews_data_Jan_Feb.json", "w") as f:
    f.write(jsonResult)
