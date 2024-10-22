from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.models.predict_model import SummaryPipeline  # Adjust as necessary

# Initialize the FastAPI app
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define model and prompt paths
model_path = 'D:\\LLUMO-AI-ASSESMENT\\models\\mart'  # Adjust as necessary
prompt_path = 'D:\\LLUMO-AI-ASSESMENT\\prompts.yaml'  # Adjust as necessary

# Initialize the SummaryPipeline
summary_pipeline = SummaryPipeline(model_dir=model_path, prompt_file=prompt_path)

class TextToSummarize(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/summarize/")
async def summarize(text_data: TextToSummarize):
    summary = summary_pipeline.generate_summary(text_data.text)
    return {"summary": summary}
