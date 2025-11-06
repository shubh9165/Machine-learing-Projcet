

import pandas as pd
import sys
import os
from src.exception import CustomExcption
import pickle
from sklearn.metrics import r2_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV
import dill
import pickle


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            mo=pickle.load(file_obj)
            return mo

    except Exception as e:
        raise CustomExcption(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomExcption(e, sys)
    
def evlauate_model(X_train,X_test,y_train,y_test,models,params):

    try:
        logging.info("Now it reporting models")
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models.keys())[i]]

            #model.fit(X_train,y_train)

            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)

            train_pred=model.predict(X_train)
            test_pred=model.predict(X_test)

            score=r2_score(y_test,test_pred)
            report[list(models.keys())[i]]=score

            return report

            
    except Exception as e:
        raise CustomExcption(e,sys)