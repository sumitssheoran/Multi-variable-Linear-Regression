import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


import sas7bdat
from sas7bdat import *

from sklearn.model_selection import train_test_split


import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.formula.api import ols

from scipy import stats
import scipy.stats as stats


# Reading dataset and creating dataframe
file=SAS7BDAT('walmart.sas7bdat')
file
df=file.to_data_frame()
df.head()
df.info()

# Check dimension of the dataset
df.shape

# Check Column names
df.columns

# Check descriptive stats of each column
m=df.describe()

# Now creating training and testing dataset from orginal walmart dataset
# 80% for training & 20% for testing
train,test=train_test_split(df,test_size=0.2,random_state=2)

# this function tells the dimensions of the data frame
train.shape
test.shape

# Seeing correlation between all variables
corr=df.corr()

#y=f(x)

x_train=train.drop('Customer_Satisfaction',axis=1)
y_train=train['Customer_Satisfaction']
x_test=test.drop('Customer_Satisfaction',axis=1)
y_test=test['Customer_Satisfaction']

# creating first model with all variables: Ordinary Least Square(OLS) Technique used
model=ols('Customer_Satisfaction ~ Product_Quality + E_Commerce + Technical_Support + Complaint_Resolution + Advertising + Product_Line + Salesforce_Image + Competitive_Pricing + Warranty_Claims + Packaging + Order_Billing + Price_Flexibility + Delivery_Speed',
          data=train).fit()

'Customer_Satisfaction' + " ~ " + " + ".join(list(x_train.columns))

# Check Model summary
model.summary()

# checking the assumption of Linear regression
# Checking multicollinearity using vif
C_df = add_constant(df)
X=C_df

g=[]
for i in range(X.shape[1]):
    g.append(variance_inflation_factor(X.values, i))
    

#Removing variables with higher VIF values(i.e Productline, Price flexibility and Delivery Speed) )

X=C_df.drop(['Delivery_Speed','Product_Line'],axis=1)

model=ols('Customer_Satisfaction ~ Product_Quality + E_Commerce + Technical_Support + Complaint_Resolution + Advertising + Salesforce_Image + Competitive_Pricing + Warranty_Claims + Packaging + Order_Billing + Price_Flexibility',
          data=train).fit()
model.summary()

# Dropping variables whose p-values are greater than 5%
model=ols('Customer_Satisfaction ~ Product_Quality + E_Commerce + Technical_Support + Complaint_Resolution + Advertising + Salesforce_Image + Competitive_Pricing + Warranty_Claims + Packaging + Order_Billing + Price_Flexibility',
          data=train).fit()
model.summary()

model=ols('Customer_Satisfaction ~ Product_Quality + E_Commerce + Technical_Support + Complaint_Resolution + Advertising + Salesforce_Image + Competitive_Pricing + Warranty_Claims + Packaging + Order_Billing',
          data=train).fit()
model.summary()

model=ols('Customer_Satisfaction ~ Product_Quality + E_Commerce + Technical_Support + Complaint_Resolution + Advertising + Salesforce_Image + Competitive_Pricing + Packaging + Order_Billing',
          data=train).fit()
model.summary()

model=ols('Customer_Satisfaction ~ Product_Quality + E_Commerce +  Complaint_Resolution +  Salesforce_Image + Competitive_Pricing + Packaging + Order_Billing',
          data=train).fit()
model.summary()

model=ols('Customer_Satisfaction ~ Product_Quality +  Complaint_Resolution +  Salesforce_Image + Competitive_Pricing + Packaging ',
          data=train).fit()
model.summary()

model=ols('Customer_Satisfaction ~ Product_Quality + Product_Line + Complaint_Resolution +  Salesforce_Image  + Packaging ',
          data=train).fit()
model.summary()

# checking vif of remaining predictors
X=C_df.drop(['Delivery_Speed','Price_Flexibility', 'Warranty_Claims', 'Complaint_Resolution', 'Advertising', 'E_Commerce'],axis=1)

pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
           index=X.columns)

# Check Residual
y_pred=model.predict(x_train)
residual=y_train-y_pred
residual_percentage = (((y_train-y_pred)/y_train)*100)#residual obtained is in percentage

# Check normality of Residuals
t=pd.DataFrame(residual)
t.describe()
stats.probplot(residual, dist="norm",plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

# 1. Checking normality of residuals
hist, bins = np.histogram(residual, bins=10)
hist
bins
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center,hist, align='center', width=width)
plt.ylabel('frequency')
plt.xlabel('residual')
plt.show()




# 3. Checking Homoscedasticity
plt.plot(y_train,residual,'o')
plt.xlabel('y_train')
plt.ylabel('residual')

## Breusch-Pagan Test
name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
bp = statsmodels.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
bp
pd.DataFrame(name,bp)

#H0: variance of residuals is const(homo condition), H1: var.of residuals not const(hetero)
#We see that the p-value is 0.32. This is higher than 0.05, so we can no longer reject the null hypothesis of homoscedasticity.
#In this regard, the Breusch-Pagan test has confirmed that our results are not influenced by heteroscedasticity and we are therefore reliable


# checking linearity
fig=sm.graphics.plot_partregress('Customer_Satisfaction','Product_Quality',['Complaint_Resolution', 'Order_Billing','Salesforce_Image'],
                                 data=df,obs_labels=False)# Effect of ['Complaint_Resolution', 'Order_Billing','Salesforce_Image'] will be removed by OLS regression.

fig1=sm.graphics.plot_partregress('Customer_Satisfaction','Complaint_Resolution',['Product_Quality', 'Order_Billing','Salesforce_Image'],data=df,obs_labels=False)
fig2=sm.graphics.plot_partregress('Customer_Satisfaction','Order_Billing',['Product_Quality','Complaint_Resolution','Salesforce_Image'],data=df,obs_labels=False)
fig3=sm.graphics.plot_partregress('Customer_Satisfaction','Salesforce_Image',['Product_Quality','Complaint_Resolution','Order_Billing'],data=df,obs_labels=False)



# Predicting the Test set results
y_pred=model.predict(x_test)

plt.plot(y_test,y_pred,'o')
plt.xlabel('Actual')
plt.ylabel('Predictions')


#Model evaluation metrics for regression-------------------------

from sklearn import metrics
print(metrics.mean_absolute_error(y_test, y_pred))#MAE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))#RMSE

0.7465290637171569/y_test.mean()

import seaborn as sns
sns.lmplot('Product_Quality','Customer_Satisfaction',train,fit_reg=True)
#fit_reg : If True, estimate and plot a regression model relating the x and y variables.
sns.lmplot('Product_Quality','Customer_Satisfaction',train,lowess=True)
#lowess : If True, use statsmodels to estimate a nonparametric lowess model (locally weighted linear regression).












