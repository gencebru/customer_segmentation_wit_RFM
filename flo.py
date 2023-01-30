import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)



df_ = pd.read_csv("dataset/flo_data_20k.csv")
df = df_.copy()

df.head(10) # ilk 10 değer
df.columns # değişken isimleri
df.shape  #betimsel istatistik
df.isnull().any() #null değer var mı?
df.dtypes  #değişken tipleri


df["order_num_total"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# last_order_date , first_order_date, last_order_date_online, last_order_date_offline bunlar pandas serisi. datetime çevirilmeli.
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])




df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total": "sum",
                                 "customer_value_total": "sum"})

df.sort_values("customer_value_total", ascending=False)[:10]

df.sort_values("order_num_total",ascending=False)[:10]

def data_prep (dataframe):
    df["order_num_total"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df


today_date = dt.datetime(2021,6,1)

rfm = pd.DataFrame()
rfm["musteri_id"] = df["master_id"]
rfm["recency"] = (today_date - df["last_order_date"])
#rfm["recency"].head()
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]
rfm.head()



rfm["recency_score"] = pd.qcut(rfm["recency"],5,labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"],5,labels=[5,4,3,2,1])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


case_a_segment_ID = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["musteri_id"]
case_a_customer_ID = df[(df["master_id"].isin(case_a_segment_ID)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
case_a_customer_ID.to_csv("case_a_musteri_ID.csv",index=False)
case_a_customer_ID.count()

case_b_segment_ID = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["musteri_id"]
case_b_customer_ID = df[(df["master_id"].isin(case_b_segment_ID)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
case_b_customer_ID.to_csv("case_b_musteri_ID.csv", index=False)
