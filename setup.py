from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project focuses on fine-tuning a pre-trained GenAI LLM model to automatically generate concise summaries of academic paper meta-reviews. Using the meta-review dataset from Hugging Face, the model is fine-tuned for summarization tasks. The final output includes a creative prompt that directs the model to generate key-point summaries. Evaluation is done using metrics like ROUGE to ensure summary quality.',
    author='Ayush Kumar',
    license='MIT',
)
