# Welcome to PAPI (Prediction API) - A powerful routing/classification tool for unstructured data.

 developed by [Christian Bernecker](https://www.linkedin.com/in/bernecker-christian-ba5ab4170/) 
 developed for: Machine Learning Engineers, Data Scientists and all other people that want to bring there prediction models in the real world


PAPI is a REST API that stands for an easy handling of classification models. PAPI can be used on your local systems for testing or on your server and the IBM Cloud for production usage. It has an integrated caching system that allows you to do prediction within MS. The only thing you have to do is to upload your models under the prediction_models folder and define it on the missionInScope.json.   

For a demo purpose the system is already feed with 2 identical models (ModelX and ModelY) based on the KAGGLE News Category DataSet: https://www.kaggle.com/rmisra/news-category-dataset. I only picked 4 classes of the dataset. 

At the moment is supports: 
+ SVC in the format *.sav from the joblib dump with protocol=2
* SGD in the format *.sav from the joblib dump with protocol=2
* MLP in the format *.hp5 from the joblib dump with protocol=2 (bugfixing at the moment)

Planned
+ k-NearstNeigbour
* Naive Bayes

If you have any idea for enhancement or you find bug please open an issue and I'll try to answer soons as possible. 

# Requirements

* Python 3.6.1 or higher
* [IBM Cloud Account](https://cloud.ibm.com/registration)
* install requirments.txt with pip -m install -r requirments.txt

## 1. Setup

This is an example and you can use it out of the box. Only execute prediction.py after you istalled all the modules that are listed in the requirments.txt. You can do that with 

```
pip install -r requirements.txt
```

then run the app with

```
python prediction.py
```

afterwards you will see in your console:

```
2019-02-06 14:53:25,144 192.168.0.1 predictionAPI Starting Server
2019-02-06 14:53:25,144 192.168.0.1 predictionAPI Caching Model MODELX
2019-02-06 14:53:26,018 192.168.0.1 predictionAPI Caching Model MODELY
2019-02-06 14:53:26,889 192.168.0.1 predictionAPI Server Ready
```

## 2. Test the app with [Postman](https://www.getpostman.com/downloads/)

Choose your API endpoint

Replace the *API-endpoint* in the command with an API endpoint from the following list.

|URL                                                  | Description    | Region         |
|-----------------------------------------------------|----------------|----------------|
| http://localhost:8000/api/v1/route                  | Localhost      | local          |


The header
  ```
Content-Type:application/json
  ```

The body:

|Key              | Description of Value                                                     |
|-----------------|--------------------------------------------------------------------------|
| `modelName`       | The modelname is defined in the missionInScope.json. To call a certain model you can use the ModelAlias or ModelName as the value      |
| `title`           | The title and the message are combined in the code so you could put either in message or titel       |
| `message`         | The message and the title are combined in the code so you could put either in message or titel       |


  ```JSON
{
  "modelName" : "MODELX",   
  "title" : "2 People Injured In Indiana School Shooting",
  "message" : "A male student, believed to be the suspect, has been detained, according to police."  
}
  ```
The result (Skill Prediction):

|Key              | Description of Value                                                     |
|-----------------|--------------------------------------------------------------------------|
| `bestScore`     | shows the confidence of the highest ranked class     |
| `bestSkill`     | shows the label of highest ranked class     |
| `modelName`     | shows the model you called in the body      |
| `pConfidence`   | this is the difference between the first and second confidence. If this value is close to the best skill you know that you prediction result is good. Is the difference higher than the prediction is not so good      |
| `results`       | List of results of all other classes (descending)      |

  ```JSON
{
    "bestScore": 0.7103054802979846,
    "bestSkill": "0",
    "modelName": "MODEL Y",
    "pConfidence": 0.5872247917240718,
    "results": [
        {
            "rank": 0,
            "score": 0.7103054802979846,
            "skill": "0"
        },
        {
            "rank": 1,
            "score": 0.12308068857391269,
            "skill": "1"
        },
        {
            "rank": 2,
            "score": 0.09131455397835446,
            "skill": "3"
        },
        {
            "rank": 3,
            "score": 0.07529927714974836,
            "skill": "2"
        }
    ]
}
```

## 3. Make it Public and install it on the CLOUD 

1. Ensure you have an account [IBM Cloud Account](https://cloud.ibm.com/registration)
2. Install the [IBM CLI Tool](https://github.com/IBM-Cloud/ibm-cloud-developer-tools)
3. Open a terminal or command line
4. Enter: ```cf login`` in the console
5. Enter: ```cf push``  in the console

You should see something linke this: 

    ````
    requested state: started
    instances: 2/2
    usage: 512M x 2 instances
    urls: papi-predictionapi-doux-flagger.mybluemix.net
    last uploaded: Wed Feb 6 17:21:09 UTC 2019
    stack: cflinuxfs2
    buildpack: https://github.com/cloudfoundry/python-buildpack

        state      since                    cpu    memory        disk         details
    #0   running    2019-02-06 06:24:39 PM   0.4%   40K of 512M   8K of 2G
    #1   starting   2019-02-06 06:24:11 PM   0.4%   40K of 512M   2.9M of 2G
    ```


Go to point 2 and use your public url to see how it works. 

## 5. Add your own model

Go to the modellinScope.json file and add your model. You have to enter the path to the models and the ModelName and the ModelAlias. This is important because you call the model via the ModelName or ModelAlias Attribute via REST. 

  ```JSON
[
    {
    "ModelName": "MODEL X",
    "ModelAlias": "MODELX",
    "ModelDescription" : "",
    "SGD": "./prediction_models/MODELX/SGDClassifier.sav",
    "SVM": "./prediction_models/MODELX/SVMClassifier.sav",
    "MLP": "./prediction_models/MODELX/MLPClassifier.h5"
  },
  {
    "ModelName": "MODEL Y",
    "ModelAlias": "MODELY",
    "ModelDescription" : "IBM Security",
    "SGD": "./prediction_models/MODELY/SGDClassifier.sav",
    "SVM": "./prediction_models/MODELY/SVMClassifier.sav"
  }
] 
```


**Important** at the moment you can only define SGD, SVM, MLP in the files. More of them comming soon. If you have a wish for a model please open an issue and let me know. 