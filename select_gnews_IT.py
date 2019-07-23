import gnews_data_Mar_Apr
import gnews_data_Mar_Apr_IT
import csv
import json

gnews_preprocessed = gnews_data_Mar_Apr.data
gnews_processed = gnews_data_Mar_Apr_IT.data

def loadCSV(filename):
    # csv obtained here;
    # https://datahub.io/core/s-and-p-500-companies#data
    compList = []
    with open(filename, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            compList.append(row[0])
    return compList

data_count = 0
#idx = 0
IT_comp_list = loadCSV("Tech_industry_list.csv")
for comp in IT_comp_list:
    comp_result = gnews_preprocessed[comp]
    #if idx < 1:
    #    print(comp)
    #    print("\n")
    #    print(comp_result[16])
    #    print("\n")
    #    print(len(comp_result))
    #    print("\n")
    print(comp)
    data_count += len(comp_result)
    print(data_count)
    gnews_processed.update({comp: comp_result})
    #idx += 1

json_data = json.dumps(gnews_processed)

with open('gnews_data_Mar_Apr_IT.json','w') as f:
    #f.write("data = ")
    f.write(json_data)

print("Final data count:" + str(data_count))
