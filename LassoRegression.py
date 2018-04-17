import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('93cars.csv')
X = np.array(df[['MaximumPrice','RPM','FuelTankCapacity','PassengerCapacity','Length','HighwayMPG','AirBagsStandard','DriveTrainType']])
Y = df['EngineSize'].values
X = sm.add_constant(X)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, Y)             # Training
pre = lasso_reg.predict(X)      # Predicting
err = np.sum((pre - Y) ** 2)
mse = np.sqrt(err/len(pre))
print "R2", lasso_reg.score(X, Y)   # Accuracy
print "Sum of squared errors = ",err
print "y = ", lasso_reg.coef_, "x + ", lasso_reg.intercept_
print

plt.scatter(pre,Y)
plt.show()
