import pandas as pd
import matplotlib.pyplot as plt

header = ['date', 'freq']
df = pd.read_csv('date_distribution.csv',names=header, index_col=False)
df.hist(column='freq')
