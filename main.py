import os
import sys
from src.models.predict_model import SummaryPipeline  # Adjust the import based on your project structure

def main():
    # Define model and prompt paths
    model_path = 'D:\\LLUMO-AI-ASSESMENT\\models\\mart'  # Adjust as necessary
    prompt_path = 'D:\LLUMO-AI-ASSESMENT\prompts.yaml'  # Adjust as necessary

    # Initialize the SummaryPipeline
    summary_pipeline = SummaryPipeline(model_dir=model_path, prompt_file=prompt_path)

    # Get user input for the text to summarize
    input_text = input("Please enter the text or meta-review to summarize:\n")

    # Generate summary
    summary = summary_pipeline.generate_summary(input_text)

    # Print the generated summary
    print("Generated Summary:", summary)

if __name__ == "__main__":
    main()
