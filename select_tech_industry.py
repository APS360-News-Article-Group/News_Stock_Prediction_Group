import csv
import pandas as pd

header = ['symbol', 'company', 'industry']
df = pd.read_csv('S&P500_list.csv',names=header, index_col=False)

relevant = pd.concat([df['industry'] == 'Information Technology'], axis=1).any(axis=1)
df_relevant = df[relevant]
print(df_relevant)

df_relevant.to_csv('Tech_industry_list.csv', encoding='utf-8', index=False, header=False)
print("Finish")
