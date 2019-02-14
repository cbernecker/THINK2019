import os
import pandas as pd
import numpy as np 

#scikit learn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

def create_SGD_Model(data):

    (train_texts, train_labels), (test_texts, test_labels) = data 
    text_svm_pipe = Pipeline([
                        ('tfidf_vector_com', TfidfVectorizer(
                                input='array', encoding="ISO-8859-1",
                                max_df=0.8,
                                min_df=1,
                                norm='l2',
                                max_features=None,  # for 1CLOSEDPDPosremoved1
                                lowercase = True,
                                sublinear_tf = False,
                                stop_words='english', 
                                analyzer ='word'
                                #ngram_range=(1,1)
                            )),
                            ('SGD', SGDClassifier(
                                    verbose=0,
                                    loss='log', 
                                    n_jobs=2, 
                                    class_weight="balanced", 
                                    penalty = "l2",  
                                    max_iter=1000, 
                                    tol = 1e-3)), 
                            ])    

    #SGD
    print("\n --- TEST OUTPUT SGD ---")
    text_svm_pipe.fit(train_texts,train_labels)
    predicted_SWR = text_svm_pipe.predict(test_texts)
    predicted_SWR_train = text_svm_pipe.predict(train_texts)
    joblib.dump(text_svm_pipe, 'SGDClassifier.sav', protocol=2)

    #Metrics
    print("Chance: " + str((100/4)/100))
    print("Accuracy SGD test data:  " + str(np.mean(predicted_SWR == test_labels)))  
    print("Accuracy SGD train data: " + str(np.mean(predicted_SWR_train == train_labels)))
    print("\n --- CONFUSTION MATRIX SGD (test data)---")
    print(metrics.confusion_matrix(test_labels, predicted_SWR))
    print("\n --- CLASSIFICATION REPORT SGD (test data)---")
    print(metrics.classification_report(test_labels, predicted_SWR))


# if __name__ == "__main__":
#     print(__name__)
#     data_path = "../data/News_Category_Dataset_v2.json"
#     create_SGD_Model(load_json_data(data_path))

