##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################################

# 1. Data Preparation
# 2. Expected Number of Transaction with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. Calculating CLTV with BG-NBD and Gamma-Gamma Model
# 5. Segmentation

# The dataset named "Online Retail II" includes the sales of an UK-based online store between 01/12/2009 - 09/12/2011.
# dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Variables
# Invoice       :Invoice number. The unique number of each transaction, namely the invoice. If it starts with C, it shows the canceled invoice
# StockCode     :Unique number for each product
# Description   :Product description
# Quantity      :It expresses how many of the products on the invoices have been sold.
# InvoiceDate   :Invoice date and time
# UnitPrice     :Product price (in GBP)
# CustomerID    :Unique customer number
# Country       :Country where the customer lives

##############################################################
# 1. Data Preparation
##############################################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df.head()
df.isnull().sum()
df.dropna(inplace=True)

df = df[~df['Invoice'].str.contains('C', na=False)]

df.describe().T

df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]

replace_with_thresholds(df, 'Quantity')
replace_with_thresholds(df, 'Price')

df.describe().T

df['TotalPrice'] = df['Quantity'] * df['Price']

today_date = dt.datetime(2011, 12, 11)

# recency: Time, since last user-specific purchase on a weekly basis
# T: how long before the first purchase was made the analysis date (the age of the customer on a weekly basis)
# frequency: total number of repeat purchases (frequency>1)
# monetary: average earnings per purchase


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                                                         lambda x: (today_date - x.min()).days],
                                         'Invoice': lambda x: x.nunique(),
                                         'TotalPrice': lambda x: x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df['recency'] = cltv_df['recency'] / 7

cltv_df['T'] = cltv_df['T'] / 7

cltv_df.describe().T

##############################################################
# 2. Expected Number of Transaction with BG-NBD Model
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# Who are the 10 customers we expect the most to purchase in 1 week?

# cltv_df.groupby('Customer ID').agg('sum').sort_values(by='frequency', ascending=False)
# cltv_df['frequency'].sort_values(ascending=False)

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df['expected_purc_1_week'] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# Who are the 10 customers we expect the most to purchase in 1 month?

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df['expected_purc_1_month'] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# What is the total number of sales in 1 month?

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

# What is the Expected Number of Sales of the Whole Company in 3 Months?

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df['expected_purc_3_month'] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# Evaluation of Forecast Results

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. Expected Average Profit with Gamma-Gamma Model
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'],
        cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values('expected_average_profit', ascending=False).head(10)

##############################################################
# 4. Calculating CLTV with BG-NBD and Gamma-Gamma Model
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,      # 3 month
                                   freq="W",    # Frequency of T value.
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on='Customer ID', how='left')

cltv_final.sort_values(by='clv', ascending=False).head(10)

##############################################################
# 5. Segmentation
##############################################################

cltv_final['segment'] = pd.qcut(cltv_final['clv'], 4, labels=['D', 'C', 'B', 'A'])

cltv_final.sort_values(by='clv', ascending=False).head(50)

cltv_final.groupby('segment').agg({'count', 'mean', 'sum'})


new_df = pd.DataFrame()
new_df['cltv_top'] = cltv_final[cltv_final['segment'] == 'A']['Customer ID']
new_df['cltv_top'] = new_df['cltv_top'].astype(int)

new_df['champions_id'] = new_df['champions_id'].astype(int)

new_df.to_csv('cltv_top_customers.csv')
