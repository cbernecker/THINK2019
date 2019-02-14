### ELIZA
from sklearn.externals import joblib    #callSkillModels
from lib.nlp.nlp import removeStopwords
import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

### BLUEMIX
from flask import Flask, render_template, request, jsonify
from cloudant import Cloudant
import atexit
import json
import datetime
import requests
import logging

### Common
import os, sys
import string, json

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'prediction'}
logger = logging.getLogger('tcpserver')
logger.setLevel(logging.INFO)

productdata = json.load(open('modelInScope.json'))


def preCacheModels():
    '''
    We load the models in a dictionary to hold them in memory. That increases
    our response time. Because loading from the file system is expensive. 
    '''
    cachedSVMModels = {}
    cachedSGDModels = {}
    for products in productdata:
        logger.info("Caching Model %s", products["ModelAlias"]  , extra=d)
        # mutable object for a lower memory footprint
        SGD_MODEL, SVM_MODEL = [], []
        try:
            SGD_PATH = os.path.abspath(os.path.join(products["SGD"]))
            SGD_MODEL = [joblib.load(SGD_PATH)]
        except FileNotFoundError:
            logger.error("No Model found in Path %s ", SGD_PATH,  extra=d )
        try:
            SVM_PATH =  os.path.abspath(os.path.join(products["SVM"]))
            SVM_MODEL = [joblib.load(SVM_PATH)]
        except FileNotFoundError:
            logger.error("No Model found in Path %s", SVM_PATH  , extra=d)
        # Cache Alias and Name for fault tolerance
        cachedSVMModels[products["ModelAlias"]] = SGD_MODEL
        cachedSGDModels[products["ModelAlias"]] = SVM_MODEL
        cachedSVMModels[products["ModelName"]] = SGD_MODEL
        cachedSGDModels[products["ModelName"]] = SVM_MODEL
    return cachedSVMModels, cachedSGDModels

# compare which of the modells has the best result and return
### put this function to lib/prediction
def check_best_modell (prob_per_class_dictionary_reg, prob_per_class_dictionary_svm):
    '''
    Acutally we have two models in place. SVM and SGD both will return different results.
    This function compares the first predicted results of both models and will retrun the
    one with the highest confidence for the first skill.
    '''
    if (prob_per_class_dictionary_reg[0][1] > prob_per_class_dictionary_svm[0][1]):
        return prob_per_class_dictionary_reg
    else:
        return prob_per_class_dictionary_svm

### put this function to lib/prediction
def callSkillModels(title, message, product): 

    #check if product is in scope
    
    if product in cachedSVMModels.keys() and product in cachedSGDModels.keys():       
        message = title + " " + message
        mytest = [message]
        try:
            model_file_sgd = cachedSGDModels[product][0] # load SGD Model
        except IndexError:
            #logger.error('No Model found for %s check the Error Logs ', product , extra=d)
            raise IndexError('No Model found for %s check the Error Logs: ', product) 
        try:
            model_file_svm = cachedSVMModels[product][0] # load SVM Model
        except IndexError:
            #logger.error('No Model found for %s check the Error Logs ', product , extra=d)
            raise IndexError('No Model found for %s check the Error Logs: ', product) 
    else:
        raise ValueError('The product name: ' + str(product) + ' is not in scope!') 
    
    # calculate SVM results
    try:
        results = model_file_svm.predict_proba(mytest)
        # sort result
        prob_per_class_dictionary_svm = sorted(zip(model_file_svm.classes_, results[0]), key=lambda x: x[1], reverse=True)
        logger.debug('SVM: %s', prob_per_class_dictionary_svm, extra=d)
    except:
        logger.error('No SVM Model found for %s check the Error Logs ', product , extra=d)
        raise  ValueError('No SVM Model found for: ' + str(product))

    # calculate SGD results
    try:
        results = model_file_sgd.predict_proba(mytest)
        # sort result
        prob_per_class_dictionary_reg = sorted(zip(model_file_sgd.classes_, results[0]), key=lambda x: x[1], reverse=True)
        logger.debug('SGD: %s', prob_per_class_dictionary_reg, extra=d)
    except:
        logger.error('No SGD Model found for %s check the Error Logs ', product , extra=d)
        raise  ValueError('No SGD Model found for: ' + str(product))
        
    # check for the best result
    prob_per_class_dictionary = check_best_modell(prob_per_class_dictionary_reg, prob_per_class_dictionary_svm)
    return prob_per_class_dictionary

def predictionresults(request, origin):
    '''
    This function calls the prediction models and retrun the output.
    currently implemented: 
    if(origin == "/api/v1/route"):
        results = callSkillModels(removeStopwords(title), removeStopwords(problem), modelName)
    elif(origin == "/api/v1/swarming"):
        results = callLinkenMissonModels(removeStopwords(title), removeStopwords(problem), modelName)
    '''
    try:
        modelName = request.json['modelName']  
        logger.info('Prediction: %s', modelName, extra=d)          
        title = request.json['title']
        problem = request.json['message']
        stopwordRemoval_title = removeStopwords(title)   
        stopwordRemoval_desc = removeStopwords(problem)         
        try: 
            logger.debug('SVM: %s', request.json, extra=d)    
            results = callSkillModels(stopwordRemoval_title, stopwordRemoval_desc, modelName)
            print(results)
            print(results[1][0])
            pconfidence = 0
            if len(results) > 1:
                pconfidence = results[0][1] - results[1][1]              
            return jsonify({'modelName': modelName, 'bestSkill' :  str(results[0][0]), 'bestScore' : results[0][1], 'pConfidence' : pconfidence,
                            'results': 
                                [
                                    {"rank" : i+j, 'skill' : str(results[i][j]), 'score' : results[i][j+1]} for i in range (0, len(results)) for j in range(len(results[i])-1)
                                ]
                            })
        except (ValueError, IndexError) as error:
            logger.error('The product name: %s is not known. Error: %s', modelName, error, extra=d)
            return jsonify({'ERROR' : str(error)}), 400
    except KeyError as error:
        logger.error(' {} is not in body use modelName, title & message'.format(error), sys.exc_info(),  extra=d )
        return jsonify({'ERROR': 'KeyError: ' + str(error) + ' is not in body use modelName, caseID, title & message'}), 400
    
app = Flask(__name__)

# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

@app.route('/')
def home():
    return render_template('index.html')

# SKILL API
@app.route('/api/v1/route', methods=['POST'])
def prediction():
    # do the prediction
    return predictionresults(request, "/api/v1/route")

if __name__ == '__main__':
    logger.info('Starting Server', extra=d)
    cachedSVMModels, cachedSGDModels = preCacheModels()
    logger.info('Server Ready', extra=d)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
