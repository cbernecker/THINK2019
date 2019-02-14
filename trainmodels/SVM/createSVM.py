import os
import pandas as pd
import numpy as np 

#scikit learn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def create_SVM_Model(data):

    (train_texts, train_labels), (test_texts, test_labels) = data 
    # Support Vectormachine
    svm_pip = Pipeline([ 
            ('tfidf_vector_com', TfidfVectorizer(
                input='array', encoding="ISO-8859-1",
                max_df=0.8,
                min_df=1,
                norm='l2',
                max_features=None,  
                lowercase = True,
                sublinear_tf = True,
                stop_words='english', 
                # vocabulary=vocabulary,
                # analyzer ='word',
                # ngram_range=(1,2)
            )),
            #('clf', LinearSVC(C=0.9, max_iter=1000, intercept_scaling=1, class_weight='balanced'))
            ('clf', SVC(C=10 , 
                cache_size=10000, 
                kernel='rbf', 
                gamma = 0.1,
                probability =True, 
                class_weight=None,
                tol=0.001)) #rbf, class_weight='balanced' C=10, cache_size=1000, kernel='rbf', gamma=1, probability =True, class_weight='balanced'
        ])   

    #SGD
    print("\n --- TEST OUTPUT SGD ---")
    svm_pip.fit(train_texts,train_labels)
    predicted_SWR = svm_pip.predict(test_texts)
    predicted_SWR_train = svm_pip.predict(train_texts)
    joblib.dump(svm_pip, 'SVMClassifier.sav', protocol=2)

    #Metrics
    print("Chance: " + str((100/4)/100))
    print("Accuracy SVM test data:  " + str(np.mean(predicted_SWR == test_labels)))  
    print("Accuracy SVM train data: " + str(np.mean(predicted_SWR_train == train_labels)))
    print("\n --- CONFUSTION MATRIX SVM (test data)---")
    print(metrics.confusion_matrix(test_labels, predicted_SWR))
    print("\n --- CLASSIFICATION REPORT SVM (test data)---")
    print(metrics.classification_report(test_labels, predicted_SWR))


