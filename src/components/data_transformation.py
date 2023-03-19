import sys

from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    prepossesor_ob_file_path=os.path.join("artifacts","preprocessor.pkl")
    
    
class DataTransformation:
    def __inti__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_data_transformer_object(self):
        """This function is used for data transformation"""
        
        try:
            numerical_columns = ["writing score","reading score"]  
            categorical_columns =[
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
                
                
            ]   
            
            num_pipeline = Pipeline(
                
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                    
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical columns  completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            
            logging.info("returned preprocessor")
            
            
            return preprocessor
        
            
        
        except Exception as e:
            
            raise CustomException(e,sys)       
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("read train and test data")
            logging.info("obtaining preprocessing object")
            
            
            preprocessing_object = self.get_data_transformer_object()
            
            target_column_name="math score"
            numerical_columns = ["writing score","reading score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = train_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframe")
            
            input_featue_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.fit_transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_featue_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            # logging.info("saved preprocessing object")
            
            save_object(
                
                file_path = DataTransformationConfig.prepossesor_ob_file_path,
                obj = preprocessing_object
             )
            
            return (
                train_arr,
                test_arr,
                DataTransformationConfig.prepossesor_ob_file_path
            )
            
            logging.info("saved")
            
            
        except Exception as e:
            raise CustomException(e,sys)
            
            
        
