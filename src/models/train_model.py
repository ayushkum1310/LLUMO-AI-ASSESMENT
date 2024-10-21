import os
import sys
import pandas as pd
import re
import yaml
from pathlib import Path
from dataclasses import dataclass
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    model_name: str = "facebook/bart-large-cnn"  # Default value
    train_data_path: str = "D:\LLUMO-AI-ASSESMENT\data\processed\processed_train.csv"  # Default train data path
    val_data_path: str = "D:\LLUMO-AI-ASSESMENT\data\processed\processed_validation.csv"  # Default validation data path
    output_dir: str = './results'
    logging_dir: str = './logs'
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 5e-5
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 2

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.tokenizer = BartTokenizer.from_pretrained(self.config.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_name)

    def preprocess_text(self, text):
        """Preprocess text by removing punctuation and converting to lowercase."""
        if isinstance(text, str):
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            return text.lower()  # Convert to lowercase
        return ''

    def load_and_preprocess_data(self):
        """Load and preprocess the training and validation datasets."""
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            val_data = pd.read_csv(self.config.val_data_path)

            # Preprocess the text in 'Input' and 'Output' columns
            train_data['Input'] = train_data['Input'].apply(self.preprocess_text)
            train_data['Output'] = train_data['Output'].apply(self.preprocess_text)
            val_data['Input'] = val_data['Input'].apply(self.preprocess_text)
            val_data['Output'] = val_data['Output'].apply(self.preprocess_text)

            return Dataset.from_pandas(train_data), Dataset.from_pandas(val_data)
        except Exception as e:
            raise CustomException(
                f"Error loading data: {e}",
                sys
            )

    def tokenize_function(self, examples):
        """Tokenization function for the datasets."""
        inputs = self.tokenizer(examples['Input'], truncation=True, padding='max_length', max_length=1024)
        targets = self.tokenizer(examples['Output'], truncation=True, padding='max_length', max_length=150)
        inputs['labels'] = targets['input_ids']
        return inputs

    def train_model(self):
        """Train the BART model on the provided datasets."""
        try:
            train_dataset, val_dataset = self.load_and_preprocess_data()

            # Tokenize datasets
            tokenized_train_dataset = train_dataset.map(self.tokenize_function, batched=True)
            tokenized_val_dataset = val_dataset.map(self.tokenize_function, batched=True)

            # Set training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                logging_dir=self.config.logging_dir,
                logging_steps=10,
                learning_rate=self.config.learning_rate,
                save_steps=self.config.save_steps,
                evaluation_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_total_limit=self.config.save_total_limit
            )

            # Define Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_val_dataset,
                tokenizer=self.tokenizer
            )

            # Train the model
            trainer.train()

            # Save the fine-tuned model
            self.model.save_pretrained(Path('D:\LLUMO-AI-ASSESMENT\models\fine_tuned_bart'))
            self.tokenizer.save_pretrained(Path('D:\LLUMO-AI-ASSESMENT\models\fine_tuned_bart'))

            logging.info("Fine-tuning complete and model saved.")
        except Exception as e:
            raise CustomException(
                f"Error during model training: {e}",
                sys
            )

    
    @staticmethod
    def load_config_from_yaml(file_path: str) -> ModelTrainerConfig:
        """Load configuration from a YAML file."""
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            
            # Ensure correct types
            config_dict['num_train_epochs'] = int(config_dict['num_train_epochs'])
            config_dict['per_device_train_batch_size'] = int(config_dict['per_device_train_batch_size'])
            config_dict['per_device_eval_batch_size'] = int(config_dict['per_device_eval_batch_size'])
            config_dict['warmup_steps'] = int(config_dict['warmup_steps'])
            config_dict['weight_decay'] = float(config_dict['weight_decay'])
            config_dict['learning_rate'] = float(config_dict['learning_rate'])
            config_dict['save_steps'] = int(config_dict['save_steps'])
            config_dict['eval_steps'] = int(config_dict['eval_steps'])
            config_dict['save_total_limit'] = int(config_dict['save_total_limit'])

            return ModelTrainerConfig(**config_dict)
    

# Usage
if __name__ == "__main__":
    config = ModelTrainer.load_config_from_yaml('D:\\LLUMO-AI-ASSESMENT\\hyper_parameter.yaml')
    model_trainer = ModelTrainer(config)
    model_trainer.train_model()
