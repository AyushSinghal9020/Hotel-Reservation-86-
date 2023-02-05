import pandas as pd # Data Processing 
import numpy as np # Array Processing
from sklearn.preprocessing import OneHotEncoder # Encodfing of Catetgorical Data
from sklearn.preprocessing import StandardScaler # Scaling of Data 
from imblearn.over_sampling import RandomOverSampler # Sampling of Data 
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # K Nearest Neighbors Classifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.metrics import classification_report # Classification Report

data = pd.read_csv("Hotel Reservations.csv")
data.drop("Booking_ID" , axis = 1 , inplace = True)

cat = []
for i in data.columns:
    if data[i].dtypes == float or data[i].dtypes == int:
        continue
    else:
        cat.append(i)

ohe = OneHotEncoder(drop = "first" , sparse = False)
new_data = ohe.fit_transform(data[cat])
half_data = data.drop(cat , axis = 1)

df = np.hstack((half_data.values , new_data))

train , test = np.split(df.sample(frac = 1) , [int(0.8 * len(df))])

def scaler(dataframe , oversampling = False):

    X = dataframe.drop(27 , axis = 1)
    Y = dataframe[27]

    sc = StandardScaler()
    sc.fit_transform(X)

    ros = RandomOverSampler()

    if oversampling:

        ros.fit_resample(X , Y)

    return X , Y
X_train , Y_train = scaler(train , oversampling = True)
X_test , Y_test = scaler(test)

model = KNeighborsClassifier()
model.fit(X_train , Y_train)
model.predict(X_test)

model_1 = LogisticRegression()
model_1.fit(X_train , Y_train)
model_1.predict(X_test)

model_2 = GaussianNB()
model_2.fit(X_train , Y_train)
model_2.predict(X_test)

print((classification_report(Y_test , model.predict(X_test))))
print((classification_report(Y_test , model_1.predict(X_test))))
print((classification_report(Y_test , model_2.predict(X_test))))
