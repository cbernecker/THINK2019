# THINK2019
This Repo presents the way I created the models for my PAPI- Prediction API. 

For a demo purpose I used the KAGGLE News Category DataSet: https://www.kaggle.com/rmisra/news-category-dataset. I only picked 4 classes of the dataset so that you see the resulzs faster. If you want to use classes go to the /data/loadjson.py and add the additional columns.

# Requirements

* Python 3.6.1 or higher
* install requirments.txt with pip -m install -r requirments.txt

# 1. Start

In the file builtmodels.py you will find the following code.

```Python
if __name__ == "__main__":
    data_path = "./data/News_Category_Dataset_v2.json"
    #load_json_data(data_path)
    print("\n --- CREATING SGD ---")
    create_SGD_Model(load_json_data(data_path))  # creates the SGD
    print("\n --- CREATING SVM ---")
    create_SVM_Model(load_json_data(data_path))  # creates the SVM
```



SVM
SGD
MLP