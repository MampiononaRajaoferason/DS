import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
import scipy
from sklearn.preprocessing import Normalizer, StandardScaler
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
print(pyodbc.drivers())
# google big query requirement
credentials_path = 'pamf-dwh-775beaae50ff.json'
project_id = 'pamf-dwh'
cnx  = pyodbc.connect('DRIVER={SQL Server};SERVER=154.120.135.138;DATABASE=PAMF;UID=bigdata;PWD=bigdata')

# importing datasets
sql = ''' SELECT * FROM `pamf-dwh.DFS_VIEW.deboursement`  ''' 
deboursement =pd.read_gbq( sql, project_id, private_key=credentials_path, dialect='standard')

deboursement.sort_values('disbursement_datetime',inplace=True)
deboursement['num_loan'] = deboursement.groupby('rContactID').cumcount()+1

sql= ''' SELECT del.loloanID, deb.account_Number , del.rContactID , del.disbursement_date, del.Montantdeloctroi, del.sum_principal_paid,del.sum_penalty_paid, del.sum_admin_paid, del.last_payement,del.day_late   FROM `pamf-dwh.DFS_VIEW.delinquency_loan` as del
JOIN `pamf-dwh.DFS_VIEW.deboursement` as deb ON deb.loloanID = del.loloanID '''
delinquant =pd.read_gbq( sql, project_id, private_key=credentials_path, dialect='standard')
delinquant = delinquant.merge(deboursement[['loloanID','num_loan']],on='loloanID')

sql = ''' SELECT  [loLoanNanoBalance]
      ,[loLoanID]
      ,[RemainingAmount]
      ,[RemainingAdminFee]
      ,[RemainingPenalty]
      ,[StartDate]
      ,[EndDate]
      ,[ActualEndDate]
      ,[DaysLate]
      ,[partialReimbAutho]
      ,[Period]
      ,[SavingPaymentUsed]
      ,[StatusAttribute]
      ,[LokiataInitMapped]
      ,[AutomaticReimbursementInProgress]
  FROM [PAMF].[dbo].[loLoanNanoBalance]
  WHERE [DaysLate] >0
  '''
allDelinquant = pd.read_sql(sql,cnx)


sql = 'SELECT * FROM `pamf-dwh.DFS_VIEW.approved`'
registered = pd.read_gbq( sql, project_id, private_key=credentials_path, dialect='standard')

delinquant= delinquant.merge(allDelinquant,left_on='loloanID',right_on='loLoanID')
delinquant['total_paid'] = delinquant[["sum_principal_paid",'sum_penalty_paid','sum_admin_paid']].sum(axis=1)
delinquant['total_remaining'] = delinquant[['RemainingAmount','RemainingAdminFee','RemainingPenalty']].sum(axis=1)
delinquant['pamf_remaining'] = delinquant[['RemainingAdminFee','RemainingPenalty']].sum(axis=1)

train = delinquant[['loloanID','rContactID','day_late' ,'num_loan' ,'total_remaining']]
registered=registered[["rContactID",'Age','Gender','OrangeId','Location']]#.head()

train=train.merge(registered,on='rContactID')

#importing the dataset
orange = pd.read_csv('./MG_01032019_195008.csv',sep='|')

def renamecols(lcol):
    dcol = {}
    for c in lcol:
        dcol[c]= c.split(';')[0].replace(' ','')
    return dcol


orange.dropna(inplace=True)
orange.rename(columns=renamecols( orange.columns ),inplace=True)

orange=orange[["usr_id", 'ca_rechrg', 'ca_erechrg' ,'ca_voice' ,'ca_sms' ,'t_cf']] 


train=train.merge(orange,left_on='OrangeId',how='left',right_on='usr_id')


for c in ['total_remaining','ca_rechrg' ,'ca_erechrg' ,'ca_voice' ,'ca_sms' ,'t_cf']:
    train[c] = np.log1p(train[c])
    
    
train.fillna(train.median(),inplace=True)    
cols = [ 'ca_rechrg' ,'ca_erechrg' ,'ca_voice' ,'ca_sms' ,'t_cf']
train.loc[(~np.isfinite(train.t_cf)) & train.t_cf.notnull()] = 0


clustering = pickle.load(open('telco_clustering_model.pkl', 'rb'))
scaling = pickle.load(open('Scaler_model.pkl', 'rb'))

X = train[cols] 
#scal = StandardScaler()
X = scaling.transform(X)

train['telco_cluster'] = clustering.predict(X)
for  c in ['ca_rechrg' ,'ca_erechrg' ,'ca_voice' ,"ca_sms" ,'t_cf', 'total_remaining'] :
    train[c] = np.exp(train[c])

    
    
train['utilisation_telco'] = train.telco_cluster.map({0:'Faible_Faible' ,1: 'Normal_PasDuTout', 2: 'Forte_Forte', 3: 'Moyenne_Moyenne'})
train['Groupe'] = train.telco_cluster.map({0:'1' ,1: '2', 2: '4', 3: '2'})

#delinquants 50-120 j de retard
list_client_a_inciter= train[(train['day_late'] >= 50) & (train['day_late'] <= 120) ]
list_client_a_inciter.loc[list_client_a_inciter['telco_cluster'] == 0, 'Groupe non radies'] = 1 
list_client_a_inciter.loc[list_client_a_inciter['telco_cluster'] == 1, 'Groupe non radies'] = 2 
list_client_a_inciter.loc[list_client_a_inciter['telco_cluster'] == 2, 'Groupe non radies'] = 4 
list_client_a_inciter.loc[list_client_a_inciter['telco_cluster'] == 3, 'Groupe non radies'] = 3 

clients_en_retards_Non_radies = list_client_a_inciter[['OrangeId','Groupe non radies']]

clients_en_retards_Non_radies.to_excel('Clients_en_retards_non_radies.xlsx', index = False)


#les radies 

list_client_radies= train[(train['day_late'] > 120) ]
list_client_radies.loc[list_client_radies['telco_cluster'] == 0, 'Groupe radies'] = 1 
list_client_radies.loc[list_client_radies['telco_cluster'] == 1, 'Groupe radies'] = 2 
list_client_radies.loc[list_client_radies['telco_cluster'] == 2, 'Groupe radies'] = 4 
list_client_radies.loc[list_client_radies['telco_cluster'] == 3, 'Groupe radies'] = 3 

clients_radies = list_client_radies[['OrangeId', 'Groupe radies']]

clients_radies.to_excel('Clients_radies.xlsx', index = False)
#########################################################

train= train[['OrangeId','Groupe']]
train.to_excel('All_grouped_clients.xlsx', index = False)


