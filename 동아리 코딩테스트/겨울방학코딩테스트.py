import pandas as pd
import numpy as np
##1-1
df= pd.read_excel('SUPERSTORE_2018-2021.xlsx')
df.head(10)
##1-2
df1=df[df["배송 기간"]>=7]
df1.count()
##1-3
print(df['제품 중분류'].unique())
##1-4
df2=df.sort_values(by=['고객명','수량'])


#####2 해결x
import datetime as dt
def time_change(data1):
    data = datetime.strptime(data1, '%H%M')
    if data >= 0601 and data <= 1e(200:
        print('오전')
    elif data >= 1201 and data <= 2100:
        print('오후')
    elif data >= 0000 and data <= 0600:
        print('심야')
    elif data >= 2101 and data <= 2400:
        print('심야')








######3-1
df3 =pd.read_csv('Diamonds Prices2022.csv')
df3.head()
df3.info()
df3.isnull().sum()
df3.describe(include ='all')
h=df3.hist(figsize=(30,20),bins=1000)

#####3-2
df3.info()
df3=df3.astype({'cut': 'category'})
df3=df3.astype({'color' : 'category'})
df3=df3.astype({'clarity':'category'})

df3.info()
######3-3
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = LabelEncoder()
df3['color'] = encoder.fit_transform(df3['color'].values)
df3
display(encoder.classes_)
df3.info()
df3['cut']
def ordinal_encoder(data,feature,feature_rank):
    
    ordinal_dict = {}
    
    for i, feature_value in enumerate(feature_rank):
        ordinal_dict[feature_value]=i+1
    
    data[feature] = data[feature].map(lambda x: ordinal_dict[x])
    
    return data
ordinal_encoder(df3, 'cut',  ['Fair', 'Good', 'Very Good','Premium', 'Ideal'])
df3['clarity'].unique()
ordinal_encoder(df3, 'clarity',  ['I1','SI2','SI1','VS2','VS1','VVS2', 'VVS1','IF'])
pd.get_dummies(df3['clarity'])
df3.info()
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
#####3=4
x=df3.drop(['price'], axis=1)
y=df3['price']
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = 0.7, shuffle = False)

######3-5
model = XGBRegressor()

model.fit(x_train, y_train)

y_pred = model.predict(X_valid)

######4-1


import json

with open("/Users/kjh1/Documents/Python Scripts/Sports_and_Outdoors_5.json", encoding='utf-8') as F:
    data=[json.loads(line) for line in F]
data = pd.DataFrame(data)
data.info()
help_split=data.helpful
review_positive=help_split.str.get(0)
review_negative=help_split.str.get(1)
data['review_positive'],data['review_negative'] = (review_positive,review_negative)
data=data.drop('helpful', axis= 1)

####4-2
data[['reviewerID','review_positive']].count()
###4-5 reviewerName만 결측치가있음
data.isnull().sum()
####4-6
data['asin'].describe()
###4-7
data1=data.dropna(axis=0)

data1.to_csv('/Users/kjh1/Desktop/sports and outdoor')
