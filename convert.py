import pandas as pd
import json

# JSON file
f = open ('out_data_batch8.json', "r")
  
# Reading from file
data = json.loads(f.read())

print(type(data))
# Iterating through the json
# list
# for key, value in data.items():
#     print(key)
#     print(value)
  
# Closing file
f.close()

# df = pd.read_json('out_data_batch1.json')
df = pd.DataFrame(data.items(), columns=['Number', 'Cluster'])
df = df.drop(['Number'], axis = 1)
print(df)
df.to_csv('out_data_batch8.csv')