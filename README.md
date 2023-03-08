# cltv_prediction_with_python

Average Order Value = Total Price / Total Transaction
Purchase Frequency = Total Transaction / Total Number of Customer
Repeat Rate = Number of Customer Making Multiple Purchases / Total Number of Customer
Churn Rate = 1 - Repeat Rate
Profit Margin = Total Price * 0.1
Customer Value = Average Order Value * Purchase Frequency
Customer Life Time Value = (Customer Value / Churn Rate) * Profit Margin

CLTV_prediction = Expected Number of Transaction * Expected Average Profit

Expected Number of Transaction    --> BG/NBD Model
Expected Average Profit           --> Gamma Gamma Submodel

CLTV_prediction = BG/NBD Model * Gamma Gamma Submodel

BG/NBD Model = Transaction Process (Buy) [Gamma] + Dropout Process (Till you die) [Beta]

## CLTV Prediction with BG-NBD and Gamma-Gamma

1. Data Preparation
2. Expected Number of Transaction with BG-NBD Model
3. Expected Average Profit with Gamma-Gamma Model
4. Calculating CLTV with BG-NBD and Gamma-Gamma Model
5. Segmentation

The dataset named "Online Retail II" includes the sales of an UK-based online store between 01/12/2009 - 09/12/2011.
dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

##Variables

*Invoice       :Invoice number. The unique number of each transaction, namely the invoice. If it starts with C, it shows the canceled invoice

*StockCode     :Unique number for each product

*Description   :Product description

*Quantity      :It expresses how many of the products on the invoices have been sold

*InvoiceDate   :Invoice date and time

*UnitPrice     :Product price (in GBP)

*CustomerID    :Unique customer number

*Country       :Country where the customer lives
