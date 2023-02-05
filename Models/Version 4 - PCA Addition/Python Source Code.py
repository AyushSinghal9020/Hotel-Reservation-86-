import pandas as pd # Data Processing 
import numpy as np # Array Processing
from sklearn.preprocessing import OneHotEncoder # Encodfing of Catetgorical Data
from sklearn.preprocessing import StandardScaler # Scaling of Data 
from imblearn.over_sampling import RandomOverSampler # Sampling of Data 
from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # K Nearest Neighbors Classifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.metrics import classification_report # Classification Report

data = pd.read_csv("Hotel Reservations.csv")
data.drop("Booking_ID" , axis = 1 , inplace = True)

num = []
cat = []
for i in data.columns:
    if data[i].dtypes == object:
      cat.append(i)
    else:
      num.append(i)

trf = FunctionTransformer(func = np.log1p)
data_num = trf.fit_transform(data[num])
data.drop(num , axis = 1 , inplace = True)
data_new = pd.concat([data , data_num] , axis = 1 , join = "inner")

thr_rtr = 690
repl_rtr = data_new["room_type_reserved"].value_counts()[data_new["room_type_reserved"].value_counts() <= thr_rtr].index
rtr = pd.get_dummies(data_new["room_type_reserved"].replace(repl_rtr , "room_type_reserved_other"))

thr_mst = 2000
repl_mst = data_new["market_segment_type"].value_counts()[data_new["market_segment_type"].value_counts() <= thr_mst].index
mst = pd.get_dummies(data_new["market_segment_type"].replace(repl_mst , "market_segment_type_other"))

data_new.drop(["room_type_reserved" , "market_segment_type"] , axis = 1 , inplace = True)
data_proc = pd.concat([rtr, mst , data_new], axis=1, join='inner')


ohe = OneHotEncoder(drop = "first" , sparse = False)
new_data = ohe.fit_transform(data_proc[["type_of_meal_plan" , "booking_status"]])
half_data = data_proc.drop(["type_of_meal_plan" , "booking_status"] , axis = 1)

df = pd.concat([data_proc.select_dtypes(exclude='object'), 
               pd.DataFrame(new_data,columns=ohe.get_feature_names_out(["type_of_meal_plan" , "booking_status"])).astype(int)], axis=1)


train , test = np.split(df.sample(frac = 1) , [int(0.8 * len(df))])

sample = train.drop("booking_status_Not_Canceled" , axis = 1)
pca_sample = PCA()
sample = pca_sample.fit_transform(sample)
print(pca_sample.explained_variance_ratio_)
plt.plot(np.cumsum(pca_sample.explained_variance_ratio_))

def scaler(dataframe , oversampling = False):
    
    x = dataframe.drop("booking_status_Not_Canceled" , axis = 1)
    Y = dataframe["booking_status_Not_Canceled"]

    sc = StandardScaler()
    ros = RandomOverSampler()
    pca = PCA(n_components = 10)

    sc.fit_transform(x)

    if oversampling:
        
        ros.fit_resample(x , Y)

    X = pca.fit_transform(x)
    X = pd.DataFrame(X)
    
    DataFrame = pd.concat([X,Y], axis=1, join='inner')
    
    return DataFrame , X , Y

X_train , Y_train = scaler(train , oversampling = True)
X_test , Y_test = scaler(test)

model = KNeighborsClassifier()
model.fit(X_train , Y_train)
model.predict(X_test)
print((classification_report(Y_test , model.predict(X_test))))

model_1 = LogisticRegression()
model_1.fit(X_train , Y_train)
model_1.predict(X_test)
print((classification_report(Y_test , model_1.predict(X_test))))

model_2 = GaussianNB()
model_2.fit(X_train , Y_train)
model_2.predict(X_test)
print((classification_report(Y_test , model_2.predict(X_test))))
