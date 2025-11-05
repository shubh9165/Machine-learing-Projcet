import os
import sys
from sklearn.metrics import r2_score
from sklearn.ensemble import(
    GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomExcption
from src.logger import logging
from src.utils import save_object,evlauate_model
from dataclasses import dataclass

@dataclass
class modelTrainerConfig:
    model_trainer_file_path=os.path.join('artifacts','model.pkl')

class modelTrainer:
    def __init__(self):
     self.model_trainer_config=modelTrainerConfig()

    
    def initate_model_trainer(self,train_arr,test_arr):
       
       try:
          logging.info("spliting the train and test data")

          X_train,y_train,X_test,y_test=(
             train_arr[:,:-1],
             train_arr[:,-1],
             test_arr[:,:-1],
             test_arr[:,-1]
          )

          models={
             "GradientBoostingRegressor":GradientBoostingRegressor(),
             "AdaBoostRegressor":AdaBoostRegressor(),
             "RandomForestRegressor":RandomForestRegressor(),
             "KNeighborsRegressor":KNeighborsRegressor(),
             "LogisticRegression":LogisticRegression(),
             "LinearRegression":LinearRegression(),
             "CatBoostRegressor":CatBoostRegressor(),
             "XGBRegressor":XGBRegressor()
          }

          listModel:dict=evlauate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)

          best_model_score=max(sorted(listModel.values()))

          best_model_name=list(listModel.keys())[list(listModel.values()).index(best_model_score)]

          best_model=models[best_model_name]

          if best_model_score<0.6:
             raise CustomExcption("There is no model",sys)
          logging.info("the best model founded")

          save_object(
             file_path=self.model_trainer_config.model_trainer_file_path,
             obj=best_model
          )

          predicted=best_model.predict(X_train)

          score=r2_score(y_train,predicted)
          return score

       except Exception as e:
           raise CustomExcption(e,sys)
