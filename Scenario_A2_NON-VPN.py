import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

time = 15

KNN_accuracy = []
DT_accuracy = []

while time <= 120:
  #upload dataset
  df = pd.read_csv(f'TimeBasedFeatures-Dataset-{time}s-NO-VPN.csv')

  #divide independent and dependent columns
  x = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values

  #handle missing values
  imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
  imputer.fit(x[:, :])
  x[:, :] = imputer.transform(x[:, :])

  #handle categorical variable
  le = LabelEncoder()
  y = le.fit_transform(y)

  #diviide into training set and test set
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)

  #feature scaling
  sc = StandardScaler()
  x_train = sc.fit_transform(x_train)
  x_test = sc.transform(x_test)

  #model building
  classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  classifier1.fit(x_train, y_train)

  classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  classifier2.fit(x_train, y_train) 

  #prediction using model
  y_pred1 = classifier1.predict(x_test)
  y_pred2 = classifier2.predict(x_test)

  KNN_accuracy.append(round(accuracy_score(y_test, y_pred1)*100,2))
  DT_accuracy.append(round(accuracy_score(y_test, y_pred2)*100,2))

  time *= 2

print(KNN_accuracy)
print(DT_accuracy)

datasets = ['15s', '30s', '60s', '120s']  # Dataset names
plt.figure(figsize=(8, 5))
plt.xlabel('timeout', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)

#for non-vpn
plt.bar(datasets, KNN_accuracy, color='skyblue')
plt.title('Non-VPN traffic using KNN', fontsize=12)

for i, value in enumerate(KNN_accuracy):
    plt.text(i, value + 1, f'{value}%', ha='center', fontsize=10)

plt.show()

# for vpn
plt.bar(datasets, DT_accuracy, color='skyblue')
plt.title('Non-VPN traffic using Decision Tree', fontsize=12)

for i, value in enumerate(DT_accuracy):
    plt.text(i, value + 1, f'{value}%', ha='center', fontsize=10)
  
plt.show()
