import requests

url = "https://api.itick.org/stock/kline?region=HK&code=700&kType=1&et=1741239240000&limit=100"

headers = {
    "accept": "application/json",
    "token": "40739b1a775c4f2a813d1f6b980f92e42607baeece8c4951b56c61a488482543"
}

response = requests.get(url, headers=headers)

print(response.text)