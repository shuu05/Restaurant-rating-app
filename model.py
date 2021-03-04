import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('zomato_df.csv')
x = df.drop('rate', axis=1)
y = df['rate']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=10)

et = ExtraTreesRegressor(n_estimators=125)
et.fit(x_train, y_train)

y_pred = m = et.predict(x_test)
print(y_pred)
