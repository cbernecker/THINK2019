import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_json_data(data_path):
    
    #data_path = "./data/support_data/News_Category_Dataset_v2.json"
    data_df = pd.read_json(data_path, lines=True)
    print(data_df.index)
    print(data_df.columns)

    some_values = ['CRIME','FOOD & DRINK', 'TECH', 'MONEY']
    data_df = data_df.loc[data_df['category'].isin(some_values)]

    #data_texts = data_df["short_description"]
    data_texts = data_df["headline"] + " " + data_df["short_description"]
    data_labels = data_df["category"]

    le = preprocessing.LabelEncoder()
    le.fit(data_labels)
    le.transform(data_labels)

    print(le.classes_)
    print(le.inverse_transform(le.transform(data_labels)))

    train_texts, test_texts, train_labels, test_labels = train_test_split(data_texts, le.transform(data_labels), test_size=0.20, random_state=42) 
    
    print ('Amount TestData: '  + str(len(train_texts)))
    print ('Amount TrainData: ' + str(len(test_texts)))
    #print("Example Text: " + str(train_texts))
    #print("Example Categorie " + str(train_labels))

    return ((train_texts, np.array(train_labels)),
           (test_texts, np.array(test_labels)))

#data_path = "./data/News_Category_Dataset_v2.json"
#load_json_data(data_path)

def load_json_data2(data_path):
    
    #data_path = "./data/support_data/News_Category_Dataset_v2.json"
    data_df = pd.read_json(data_path, lines=True)
    print(data_df.index)
    print(data_df.columns)

    some_values = ['CRIME','FOOD & DRINK', 'TECH', 'MONEY']
    data_df = data_df.loc[data_df['category'].isin(some_values)]

    #data_texts = data_df["short_description"]
    data_texts = data_df["short_description"]
    data_labels = data_df["category"]

    # le = preprocessing.LabelEncoder()
    # le.fit(data_labels)
    # le.transform(data_labels)

    # train_texts, test_texts, train_labels, test_labels = train_test_split(data_texts, le.transform(data_labels), test_size=0.20, random_state=42) 
    
    # print ('Amount TestData: '  + str(len(train_texts)))
    # print ('Amount TrainData: ' + str(len(test_texts)))
    # print("Example Text: " + str(train_texts))
    # print("Example Categorie " + str(train_labels))

    return data_texts, data_labels

def get_num_classes(labels):
    """Gets the total number of classes.

    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)

    # Returns
        int, total number of classes.

    # Raises
        ValueError: if any label value in the range(0, num_classes - 1)
            is missing or if number of classes is <= 1.
    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
                            missing_classes=missing_classes,
                            max_class=num_classes - 1))

    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes