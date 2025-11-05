import pandas as pd
import sys
import os
from src.exception import CustomExcption
import pickle
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomExcption(e, sys)
    
def evlauate_model(X_train,X_test,y_train,y_test,models):

    try:
        logging.info("Now it reporting models")
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            model.fit(X_train,y_train)

            train_pred=model.predict(X_train)
            test_pred=model.predict(X_test)

            score=r2_score(y_test,test_pred)
            report[list(models.keys())[i]]=score

            return report

            
    except Exception as e:
        raise CustomExcption(e,sys)