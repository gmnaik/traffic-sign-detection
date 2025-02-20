import os
import sys
import torch
import torch_directml
import time
from ultralytics import YOLO
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,load_object

@dataclass
class ModelTrainerConfig:
    yaml_filepath = os.path.join("trafficsigndetection.yaml")
    model_size = "yolov8s.pt"
    epochs=1
    batch_size=-1
    learning_rate=0.005
    optimizer = 'auto'
    img_size=640
    
    ''' 
    Epochs: 10, 50, 100
    Batch sizes: 8, 16, 32, 64
    Initial learning rates (lr0): 0.001, 0.0003, 0.0001
    Dropout: 0.15, 0.25
    Optimizer: Adam, SGD, and auto. 
    
    We shall obtain best hyperparameters using code like below:
    # Iterate through all combinations of hyperparameters
    for epochs, batch_size, lr, momentum in product(epochs_list, batch_size_list, learning_rate_list, momentum_list):
        print(f"Training with epochs={epochs}, batch_size={batch_size}, lr={lr}, momentum={momentum}")
        
        model = YOLO("yolov8s.pt")  # Load model
        
        model.train(
            data="data.yaml",
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            lr0=lr,
            momentum=momentum,
            device="cuda"
        )                     
    '''
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluatevaliddata(self,bestmodelpath):
        Valid_model = YOLO(bestmodelpath)

        # Evaluating the model on the validset
        metrics = Valid_model.val(split = 'val')
        
        print("precision(B): ", metrics.results_dict["metrics/precision(B)"])
        print("metrics/recall(B): ", metrics.results_dict["metrics/recall(B)"])
        print("metrics/mAP50(B): ", metrics.results_dict["metrics/mAP50(B)"])
        print("metrics/mAP50-95(B): ", metrics.results_dict["metrics/mAP50-95(B)"])
        
        return metrics.results_dict["metrics/precision(B)"],metrics.results_dict["metrics/recall(B)"],metrics.results_dict["metrics/mAP50(B)"], metrics.results_dict["metrics/mAP50-95(B)"]
        
        
    
    def initiate_model_trainer(self):
        try:
            logging.info("Split training and test input data")
            model = YOLO(self.model_trainer_config.model_size)
            
            model.train(
                data= self.model_trainer_config.yaml_filepath,  # Path to the dataset YAML file
                epochs=self.model_trainer_config.epochs,  # Number of training epochs
                batch=self.model_trainer_config.batch_size,  # Batch size
                imgsz=self.model_trainer_config.img_size,  # Input image size
                lr0=self.model_trainer_config.learning_rate,  # Initial learning rate
                optimizer = self.model_trainer_config.optimizer, #Optimizer to update weights
                device="cpu",  # Use GPU (set to 'cpu' if GPU is not available)
                project="runs/detect",  # Set project path
                name="train",  # Always use 'train' as the folder name
                exist_ok=True  # Overwrite existing folder instead of creating new ones
            )
            
            print("Training complete! Model saved in the YOLO runs directory.")
            
            cwd = os.getcwd()
            
            bestmodelpath = cwd + "/runs/detect/train/weights/best.pt" 
            print("bestmodelpath:",bestmodelpath)
            
            a,b,c,d = self.evaluatevaliddata(bestmodelpath)
            
            
            return "Model training completed"
        
        except Exception as e:
            raise CustomException(e,sys)
        
    