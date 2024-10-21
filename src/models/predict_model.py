from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

# Load the fine-tuned model and tokenizer for BART
fine_tuned_model = BartForConditionalGeneration.from_pretrained('D:\LLUMO-AI-ASSESMENT\models\mart')
fine_tuned_tokenizer = BartTokenizer.from_pretrained('D:\LLUMO-AI-ASSESMENT\models\mart')

# Create a summarization pipeline
summarizer = pipeline('summarization', model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Define the prompt
def generate_summary(meta_review_text):
    prompt = f"""
    Summarize the following paper meta-review in a concise and informative manner:

    Meta-Review: {meta_review_text}

    Summary:
    """
    inputs = fine_tuned_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = fine_tuned_model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    return fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage with a sample meta-review
meta_review = "The paper explores new neural network architectures for image classification but lacks proper evaluation on diverse datasets, making it hard to generalize the results."
summary = generate_summary(meta_review)
print("Generated Summary:", summary)
