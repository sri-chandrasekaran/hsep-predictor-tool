import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def svm_classifier(krnl,X_train,X_test,y_train,y_test,exp):
    clf = SVC(krnl='rbf', gamma = 'scale')
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(exp, 'accuracy score:', accuracy_score(y_test,y_pred))
    return y_pred 

def pca_svm_classifier(krnl, X_train,X_test,y_train,y_test, pca_start, pca_end, exp):
    for nc in range(pca_start, pca_end):
        pca=PCA(n_components=nc)
        pca.fit(X_train)
        X_t_train = pca.transform(X_train)
        X_t_test = pca.transform(X_test)

        clf = SVC(krnl='rbf', gamma = 'scale')
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        print(exp, 'accuracy score:', accuracy_score(y_test,y_pred))
        return y_pred 

def standardize_and_split(X):
    scaler = StandardScaler()
    scaler.fit(X)

    StandardScaler(copy=True,with_mean=True,with_std=True)
    scaled_data=scaler.transform(X)
    a=pd.DataFrame(scaled_data)

    X_test, X_train,y_test, y_train = train_test_split(scaled_data,np.ravel(y), test_size=0.20,random_state=7)
    return [X_train, X_test, y_test, y_train]

