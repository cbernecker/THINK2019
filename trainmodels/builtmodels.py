from data.loadJson import load_json_data
from SGD.createSGD import create_SGD_Model
from SVM.createSVM import create_SVM_Model
from MLP.createMLP import train_ngram_model

if __name__ == "__main__":
    data_path = "./data/News_Category_Dataset_v2.json"
    print("\n --- CREATING SGD ---")
    create_SGD_Model(load_json_data(data_path))
    print("\n --- CREATING SVM ---")
    create_SVM_Model(load_json_data(data_path))
    print("\n --- CREATING MLP ---")
    train_ngram_model(load_json_data(data_path))