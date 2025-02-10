from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_trainer import ModelTrainer
from TextSummarizer.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        # Creating an object of the ConfigurationManager class  
        config = ConfigurationManager()  

        # Calling the get_model_trainer_config() method from the ConfigurationManager object  
        # This retrieves the model training configuration  
        model_trainer_config = config.get_model_trainer_config()  

        # Creating an object of the ModelTrainer class and passing the retrieved configuration as an argument  
        model_trainer_config = ModelTrainer(config=model_trainer_config)  

        # Calling the train() method from the ModelTrainer object to start the model training process  
        model_trainer_config.train() 