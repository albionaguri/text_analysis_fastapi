from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from summarization_hf import summarize_text
from ner_hf import perform_ner
from sentiment_classification_hf import sentiment_analysis

app = FastAPI()

# Pydantic model for API request validation
class NERRequest(BaseModel):
    text: str = None
    file: str = None
    threshold: float = None
    print_result: bool = False

class SummarizationRequest(BaseModel):
    text: str = None
    file: str = None
    threshold: float = None
    print_result: bool = False

class SentimentRequest(BaseModel):
    text: str = None
    file: str = None
    threshold: float = None
    print_result: bool = False


@app.post("/ner")
def ner(request: NERRequest):
    return perform_ner(request)

@app.post("/summarization")
def summarize(request: SummarizationRequest):
    return summarize_text(request)
    
@app.post("/sentiment")
def sentiment(request: SentimentRequest):
    return sentiment_analysis(request)
   
