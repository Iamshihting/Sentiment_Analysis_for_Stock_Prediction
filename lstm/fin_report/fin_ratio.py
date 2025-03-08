import requests
import pandas as pd
import json
import time
from pprint import pprint

seasion_List = [
    '2013Q1',
    '2013Q2',
    '2013Q3',
    '2013Q4',
    '2014Q1',
    '2014Q2',
    '2014Q3',
    '2014Q4',
    '2015Q1',
    '2015Q2',
    '2015Q3',
    '2015Q4',
    '2016Q1',
    '2016Q2',
    '2016Q3',
    '2016Q4',
    '2017Q1',
    '2017Q2',
    '2017Q3',
    '2017Q4',
    '2018Q1',
    '2018Q2',
    '2018Q3',
    '2018Q4',
    '2019Q1',
    '2019Q2',
    '2019Q3',
    '2019Q4',
    '2020Q1',
    '2020Q2',
    '2020Q3',
    '2020Q4',
    '2021Q1',
    '2021Q2',
    '2021Q3',
    '2021Q4',
    '2022Q1',
    '2022Q2',
    '2022Q3',
    '2022Q4',
    '2023Q1',
    '2023Q2',
    '2023Q3'
]

items_dict = {
    '流動比率':'CurrentRatio',
    '速動比率':'QuickRatio',
    '利息保障倍數':'InterestCoverage',
    '應收款項收款率':'AccountsReceivableTurnover',
    '平均收現日數':'AccountsReceivableTurnoverDay',
    '存貨周轉率':'InventoryTurnover',
    '平均銷貨日數':'InventoryTurnoverDay',
    '總資產周轉率':'TotalAssetTurnover',
    '毛利率':'GrossMargin',
    '營業利益率':'OperatingMargin',
    '稅後純益率':'NetIncomeMargin',
    '資產報酬率':'ROA',
    '權益報酬率':'ROE',
    '營業收入年增率':'RevenueYOY',
    '營業毛利年增率':'GrossProfitYOY',
    '營業利益年增率':'OperatingIncomeYOY',
    '稅後純益年增率':'NetProfitYOY',
    '每股盈餘年增率':'EPSYOY',
    '營業現金對流動負債比率':'OperatingCashflowToCurrentLiability',
    '營業現金對負債比':'OperatingCashflowToLiability',
    '營業現金對稅後純益比':'OperatingCashflowToNetProfit',
    '負債比率':'DebtRatio',
    '長期資金佔不動產、廠房及設備比率':'LongTermLiabilitiesRatio',
    '營業收入':'Revenue',
    '營業毛利':'GrossProfit',
    '營業利益':'OperatingIncome',
    '稅後純益':'NetProfit',
    '每股盈餘':'EPS',
    '股本':'CommonStock',
    '歸屬於母公司業主之權益':'Equity',
    '每股淨值':'NAV',
    '營業活動現金流量':'OperatingCashflow',
    '投資活動現金流量':'InvestingCashflow',
    '籌資活動現金流量':'FinancingCashflow'
}

def req_twse(item):

    url = "https://mopsfin.twse.com.tw/compare/data"

    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "zh-TW,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "sec-ch-ua": "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\", \"Microsoft Edge\";v=\"120\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest"
    }

    payload = {
        "compareItem": item,
        "quarter": "true",
        # "ylabel": "%25",
        "ys": "0",
        "revenue": "true",
        "bcodeAvg": "true",
        "companyAvg": "true",
        "qnumber": "",
        "companyId": [
            "2303 聯電",
            "3711 日月光投控",
            "2454 聯發科",
            "2330 台積電",
            "2308 台達電"
        ]
    }

    response = requests.post(url, headers=headers, data=payload, cookies=None)

    return response.json()


def mk_data(item_name, res, data):
    for c in range(5):
        c_name = res['graphData'][c]['label']
        queue = {}
        for d in res['graphData'][c]['data']:
            if '年增率' in item_name:
                queue[seasion_List[d[0]+4]] = d[1]
            else:
                queue[seasion_List[d[0]]] = d[1]
        s = pd.Series(queue, name=item_name, index=seasion_List)
        
        if c_name not in data:data[c_name] = {}
        data[c_name].update({
            item_name:s
        })

data = {}
for item_name, item in items_dict.items():
    res = req_twse(item)
    mk_data(item_name, res, data)

df_list = {}
for company, d in data.items():
    df_list[company] = pd.DataFrame(d)

writer = pd.ExcelWriter(r'fin_report/fundamental.xlsx')
for c_name, c_df in df_list.items():
    c_df.to_excel(writer, sheet_name=c_name, index=True)
writer._save()


