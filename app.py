from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize app
app = FastAPI(
    title="ChatGPT Sentiment Analysis API",
    description="🚀 Analyze text sentiment using LSTM and BERT tokenizer",
    version="1.0.0"
)

# Allow all CORS (useful for HTML page access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = tf.keras.models.load_model("sentiment_lstm_model.keras")  # Make sure this file exists

# Label mapping
label_map = {0: "good", 1: "bad", 2: "neutral"}

# Pydantic model for POST input
class TextRequest(BaseModel):
    text: str = Field(..., example="ChatGPT is awesome!")

# Preprocessing function
def preprocess(texts, max_len=128):
    encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len)
    return np.array(encodings['input_ids'])

# Home route
@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to the ChatGPT Sentiment Analysis API!",
        "hint": "Visit /docs for Swagger UI or /ui for the HTML frontend."
    }

# Prediction endpoint
@app.post("/predict", summary="Predict Sentiment", tags=["API"])
def predict_sentiment(data: TextRequest):
    input_ids = preprocess([data.text])
    preds = model.predict(input_ids)
    label = np.argmax(preds, axis=1)[0]
    sentiment = label_map[label]
    return JSONResponse(content={"text": data.text, "sentiment": sentiment})

# Serve simple HTML frontend at /ui
@app.get("/ui", response_class=HTMLResponse, tags=["Frontend"])
def sentiment_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Sentiment Checker</title>
    </head>
    <body>
      <h2>ChatGPT Sentiment Analysis</h2>
      <textarea id="text" rows="6" cols="50" placeholder="Type something..."></textarea><br><br>
      <button onclick="checkSentiment()">Check Sentiment</button>
      <h3 id="result"></h3>

      <script>
        function checkSentiment() {
          const text = document.getElementById("text").value;
          fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
          })
          .then(response => response.json())
          .then(data => {
            document.getElementById("result").innerText = "Sentiment: " + data.sentiment;
          })
          .catch(err => {
            document.getElementById("result").innerText = "Error contacting the server.";
          });
        }
      </script>
    </body>
    </html>
    """
