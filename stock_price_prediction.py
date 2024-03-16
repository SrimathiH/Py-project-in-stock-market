import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

amazon = quandl.get("WIKI/AMZN")
print(amazon.head())

amazon = amazon[['Adj. Close']]
print(amazon.head())

forecast_len=30
amazon['Predicted'] = amazon[['Adj. Close']].shift(-forecast_len)
print(amazon.tail())

x=np.array(amazon.drop(['Predicted'],1))
#DataFlair - Remove last 30 rows
x=x[:-forecast_len]
print(x)

y=np.array(amazon['Predicted'])
y=y[:-forecast_len]
print(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1) 
svr_rbf.fit(x_train,y_train)

svr_rbf_confidence=svr_rbf.score(x_test,y_test)
print(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")


lr=LinearRegression()
lr.fit(x_train,y_train)

lr_confidence=lr.score(x_test,y_test)
print(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")



#Support Vector Regression (SVR) and Linear Regression.
