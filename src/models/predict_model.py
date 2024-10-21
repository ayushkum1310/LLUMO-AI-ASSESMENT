import os
import sys
import pandas as pd
import yaml  # Import PyYAML
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from src.exception import CustomException
from src.logger import logging
from src.utils.common import load_object

class SummaryPipeline():
    def __init__(self, model_dir: str, prompt_file: str) -> None:
        self.model_dir = model_dir
        self.prompt_template = self.load_prompt(prompt_file)
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.summarizer = pipeline('summarization', model=self.model, tokenizer=self.tokenizer)

    def load_prompt(self, prompt_file: str) -> str:
        try:
            with open(prompt_file, 'r') as file:
                prompt_data = yaml.safe_load(file)
            return prompt_data['prompt_template']
        except Exception as e:
            raise CustomException(f"Error loading prompt from {prompt_file}: {e}", sys)

    def load_model(self):
        try:
            return BartForConditionalGeneration.from_pretrained(self.model_dir)
        except Exception as e:
            raise CustomException(f"Error loading model: {e}", sys)

    def load_tokenizer(self):
        try:
            return BartTokenizer.from_pretrained(self.model_dir)
        except Exception as e:
            raise CustomException(f"Error loading tokenizer: {e}", sys)

    def generate_summary(self, meta_review_text: str) -> str:
        try:
            prompt = self.prompt_template.format(meta_review=meta_review_text)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise CustomException(f"Error generating summary: {e}", sys)

# Example usage with a sample meta-review (unit testing)
if __name__ == "__main__":
    model_path = 'D:\\LLUMO-AI-ASSESMENT\\models\\mart'
    prompt_path = 'DD:\LLUMO-AI-ASSESMENT\prompts.yaml'  # Adjust path as necessary
    summary_pipeline = SummaryPipeline(model_dir=model_path, prompt_file=prompt_path)

    meta_review = "The paper explores new neural network architectures for image classification but lacks proper evaluation on diverse datasets, making it hard to generalize the results."
    summary = summary_pipeline.generate_summary(meta_review)
    print("Generated Summary:", summary)
