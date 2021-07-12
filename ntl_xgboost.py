import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

data_path:str = 'C:/work/mooving/Predictive Analytics/NTL/xgboost/'
training = pd.read_csv(data_path + 'trainingMDQ_ALL.csv')
training.set_index('cuenta', inplace=True)
training.replace([np.inf, -np.inf], np.nan, inplace=True)
training.drop(['moroso','alquila','AGUSBONIFI','AGUSTARDIF','AGUSDBTAUT','AGUSDEBITO','AGUSCREDIT','grupo_ruta'], axis=1, inplace=True)
training.dropna(inplace=True)
y = training['fraude']
del training['fraude']
del training['categoria']
X_train, X_test, y_train, y_test = train_test_split(training, y, test_size=0.33, random_state=42)

xg_reg = xgb.XGBClassifier(objective ='reg:logistic', scale_pos_weight=25, colsample_bytree=0.3, max_depth=5, alpha=10, n_estimators=10, enable_categorical=True)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
predictions = [round(value) for value in preds]
rmse = np.sqrt(mean_squared_error(y_test, preds))
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))
print("RMSE: %f" % (rmse))
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)

disp = plot_precision_recall_curve(xg_reg, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: ')

plt.show()