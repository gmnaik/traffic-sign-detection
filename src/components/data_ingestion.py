import os
import sys
import pandas as pd
import cv2
import yaml
from pathlib import Path

#sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
''' 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
''' 

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    root_dir = "artifacts/data/"
    valid_formats = [".jpg", ".jpeg", ".png", ".txt"]

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.root_dir = self.ingestion_config.root_dir
        self.valid_formats = self.ingestion_config.valid_formats
        
        
    def file_paths(self,root, validformat):
        file_paths = []

        # loop over the directory tree
        for dirpath, dirnames, filenames in os.walk(root):
            # loop over the filenames in the current directory
            for filename in filenames:
                # extract the file extension from the filename
                extension = os.path.splitext(filename)[1].lower()

                # if the filename has a valid extension we build the full 
                # path to the file and append it to our list
                if extension in validformat:
                    file_path = os.path.join(dirpath, filename)
                    file_paths.append(file_path)

        return file_paths
    
    
    def write_to_file(self,images_path,labels_path,X):

        # Create the directories if they don't exist
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

        # loop over the image paths
        for img_path in X:
            # Get the image name and extension
            imgnamelist = img_path.split("/")[-1].split(".")[:-1]
            img_name = ".".join(imgnamelist)
            img_ext = img_path.split("/")[-1].split(".")[-1]
            
            # read the image
            image = cv2.imread(img_path)
            # save the image to the images directory
            cv2.imwrite(f"{images_path}/{img_name}.{img_ext}", image)

            # open the label file and write its contents to the new label file
            f = open(f"{labels_path}/{img_name}.txt", "w")
            label_file = open(f"{self.root_dir}/labels/{img_name}.txt", "r")
            f.write(label_file.read())
            f.close()
            label_file.close()
            
        
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            image_path = self.ingestion_config.root_dir + "images"
            image_validformat = self.ingestion_config.valid_formats[:3]
            label_path = self.ingestion_config.root_dir + "labels"
            label_validformat = self.ingestion_config.valid_formats[-1]

            logging.info("Obtain image and label paths from where data is present")
            image_paths = self.file_paths(image_path,image_validformat)
            label_paths = self.file_paths(label_path,label_validformat)
            logging.info("Image and label paths generation is successful")
            
            logging.info("Train/Test split initiated")
            X_train, X_val_test, y_train, y_val_test = train_test_split(image_paths, label_paths, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.7, random_state=42)
            logging.info("Train/Test split is completed")
            
            X_train_normalized_paths = [str(Path(p).as_posix()) for p in X_train]
            X_test_normalized_paths = [str(Path(p).as_posix()) for p in X_test]
            X_val_normalized_paths = [str(Path(p).as_posix()) for p in X_val]
            
            train_images_path = "artifacts/data/train/images"
            train_labels_path = "artifacts/data/train/labels"
            test_images_path = "artifacts/data/test/images"
            test_labels_path = "artifacts/data/test/labels"
            valid_images_path = "artifacts/data/valid/images"
            valid_labels_path = "artifacts/data/valid/labels"
            
            logging.info("Copying of Images and labels for training,validation and testing is initiated")
            
            self.write_to_file(train_images_path,train_labels_path, X_train_normalized_paths)
            logging.info("Images and labels for training is successfully stored in artifacts/data/train folder")
            
            self.write_to_file(test_images_path, test_labels_path, X_val_normalized_paths)
            logging.info("Images and labels for validation is successfully stored in artifacts/data/valid folder")
            
            self.write_to_file(valid_images_path, valid_labels_path,X_test_normalized_paths)
            logging.info("Images and labels for test is successfully stored in artifacts/data/test folder")
            
            logging.info("Copying of Images and labels for training,validation and testing is completed")
            
            return "data ingestion completed"
        
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    logging.info("Data Ingestion started")
    obj=DataIngestion()
    ingestion_str = obj.initiate_data_ingestion()
    logging.info("Data Ingestion Completed")
    
    if ingestion_str == "data ingestion completed":
        modeltrainer = ModelTrainer()
        trainer_str = modeltrainer.initiate_model_trainer()
        
        
    ''' 
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer = ModelTrainer()
    accuracy_test_data = modeltrainer.initiate_model_trainer(train_arr,test_arr)
    '''
    print("trainer_str:",trainer_str)
    
    
     
    
