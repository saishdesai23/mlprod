import json
import requests
import pandas as pd

"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

"""Reading test batch
"""
df = pd.read_csv('data/test.csv', encoding="utf-8-sig")
df = df.head()

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')

"""POST <url>/predict
"""
resp = requests.post("http://0.0.0.0:81/predict", \
                    data = json.dumps(data),\
                    headers= header)

resp.status_code

"""The final response we get is as follows:
"""
print(resp.json())