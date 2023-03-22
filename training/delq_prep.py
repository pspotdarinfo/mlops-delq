from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace

from azureml.data.datapath import DataPath

import os
import argparse
import joblib
import pickle
import json
import pandas as pd
#from train import split_data, train_model, get_model_metrics


from azureml.core import Workspace, Datastore, Dataset
datastore_name = 'workspaceblobstore'# get existing workspace
workspace = Workspace.from_config()
# retrieve an existing datastore in the workspace by name
datastore = Datastore.get(workspace, datastore_name)

df = pd.read_csv('azureml://subscriptions/2cc8eda2-a29a-4717-9b66-db143b059be4/resourcegroups/rg-bc-az-mlops/workspaces/mlops-wsh-aml-1/datastores/workspaceblobstore/paths/insurance/delq.csv')




df = dataset.to_pandas_dataframe()
df=df.copy()
df1['PaymentID']=df['PaymentID'].fillna('ID Missing')
df = df[df.PaymentID != 'ID Missing']
df['ChargeAmount']=df['ChargeAmount'].apply(lambda x:np.log(x+1))
df['SalesTax']=df['SalesTax'].apply(lambda x:np.log(x+1))
df['AdminFee']=df['AdminFee'].apply(lambda x:np.log(x+1))
df['InvoicePayAmount']=df['InvoicePayAmount'].apply(lambda x:np.log(x+1))
df['Daylag']=df['DateTransferDoF']-df['InvoiceDate']
for i in range(0, len(df)):
    

    df.iloc[i,14]=df.iloc[i,14].days
df['Daylag']=df['Daylag'].astype('Int64')
df['Daylag']=df['Daylag'].fillna(291)
df['year']=np.nan
for i in range(0, len(df)):
    df.iloc[i,15] = df.iloc[i,11].year
df['month']=np.nan
for i in range(0, len(df)):
    df.iloc[i,16] = df.iloc[i,11].month
df=df.drop(columns=['InvoiceSequenceID','InvoiceBillAmount','PaymentID','InvoiceDate','DateTransferDoF'])
df=df.drop(columns=['InvoicePayAmount','SalesTax','AdminFee'])
df=df.drop(columns=['month'])
df=df.drop(columns=['OMONumber'])
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dfinvoptcpy2=df.copy()
dfinvoptcpy2['InvoiceStatus']= label_encoder.fit_transform(dfinvoptcpy2['InvoiceStatus'])

dfinvoptcpy2['DaysDel']= label_encoder.fit_transform(dfinvoptcpy2['DaysDel'])
dfinvoptcpy2['Payment Term']= label_encoder.fit_transform(dfinvoptcpy2['Payment Term'])

dfinvoptcpy2['year']= label_encoder.fit_transform(dfinvoptcpy2['year'])
ncols=['ChargeAmount','Daylag']

for i in ncols:
    scale= StandardScaler().fit(dfinvoptcpy2[[i]])
    dfinvoptcpy2[i]=scale.transform(dfinvoptcpy2[[i]])
dfinvoptcpy2=dfinvoptcpy2[['InvoiceID','ChargeAmount','Payment Term','InvoiceStatus','Daylag','year','DaysDel']]









from azure.storage.blob import BlobClient

storage_connection_string="DefaultEndpointsProtocol=https;AccountName=mlopspjj1amlsa;AccountKey=xk6IsEmVOlenzXOOPpBV+mUukrl2yhXsjEo3JxJFBnuLehmQaMri0LZXQ8LU3isGenaZ+AF8dOFK+AStvV7Htw==;EndpointSuffix=core.windows.net"
container_name = "azureml-blobstore-425a3630-be58-4693-b227-d526b82dbbf8"
dest_file_name = "preproc_delq.csv"

blob_client = BlobClient.from_connection_string(storage_connection_string,container_name,dest_file_name)