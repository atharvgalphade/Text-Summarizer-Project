from fastapi import FastAPI  # Imports FastAPI framework to create a web API
import uvicorn  # Imports Uvicorn, an ASGI server to run FastAPI applications
import sys  # Provides system-specific parameters and functions
import os  # Allows interaction with the operating system (e.g., file paths, environment variables)
from fastapi.templating import Jinja2Templates  # Enables rendering HTML templates using Jinja2 in FastAPI
from starlette.responses import RedirectResponse  # Allows sending HTTP redirects as responses
from fastapi.responses import Response  # Base response class for sending custom responses
from TextSummarizer.pipeline.prediction import PredictionPipeline  # Imports the prediction pipeline for text summarization


text:str="What is text summarization?"

app=FastAPI()

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py") # this os.system will run anything written in it as a command
        return Response("Training Successful!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text):
    try:
        obj= PredictionPipeline()
        text=obj.predict(text)
        return text
    except Exception as e:
        raise e
    
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0", port=8080)    