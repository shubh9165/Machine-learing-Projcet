import pandas as pd
import numpy as np
import sys
from src.exception import CustomExcption
from src.utils import load_object
import os


class predictpipeline:
    def __init__(self):
        pass

    def predict(self,features):
        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
        model=load_object(model_path)
        processor=load_object(preprocessor_path)

        sacled_data=processor.transform(features)

        pred=model.predict(sacled_data)

        return pred

        
class CustomData():
    def __init__(self,gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_data_freame(self):
        
        try:
            custom_input_data={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }

            return pd.DataFrame(custom_input_data)
        except Exception as e:
             raise CustomExcption(e,sys)