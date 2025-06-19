# 🤖 ChatGPT Sentiment Analysis API

This project is a complete **sentiment analysis pipeline** using a custom-trained **LSTM model** with **BERT tokenization**. It classifies text into three sentiment categories:

- ✅ `good`
- ⚠️ `neutral`
- ❌ `bad`

The model is trained on a Kaggle dataset of ChatGPT-related tweets and deployed using **FastAPI**, featuring:

- 🧪 **Swagger UI** for API testing
- 🌐 **Custom HTML frontend** for live user interaction

---

## 🚀 Features

- 🔤 **BERT Tokenizer** to handle natural language tokenization
- 🧠 **LSTM Neural Network** built with Keras/TensorFlow
- ⚖️ **Class-balanced training** to improve accuracy on underrepresented classes
- 🌐 Deployed as a **FastAPI service**
- 🖥️ Interactive:
  - `/predict`: API for predictions
  - `/docs`: Swagger UI for testing
  - `/ui`: HTML UI for simple input/output interface

---

## 🧠 Model Architecture

> Trained for 5 epochs using the [ChatGPT Sentiment Dataset](https://www.kaggle.com/datasets/charunisa/chatgpt-sentiment-analysis)

### 🔧 Architecture:
- Tokenizer: `bert-base-uncased` (from HuggingFace)
- Embedding layer (`input_dim=vocab_size, output_dim=128`)
- LSTM Layer: `128 units`
- Dense Layer: Softmax (3 output classes)

### 🧪 Evaluation:
- **Loss**: sparse categorical crossentropy
- **Metrics**: accuracy, precision, recall, F1-score

---

## 📁 Project Structure

📦 Chat_Gpt_Sentiment_Analysis
├── app.py # FastAPI application (includes model and HTML UI)
├── sentiment_lstm_model.keras # Saved trained LSTM model
├── requirements.txt # All dependencies for the project
├── README.md # Project documentation
└── kaggle.json # (Optional) Kaggle API key to download dataset

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/chatgpt-sentiment-analysis.git
cd chatgpt-sentiment-analysis

3️⃣ Install Dependencies
bash

pip install -r requirements.txt

▶️ Run the API
bash
py -3.10 -m uvicorn app:app --reload

🌐 Access the Interfaces
Interface	URL	Description
Swagger Docs	http://127.0.0.1:8000/docs	API testing & schema (auto-generated)
HTML Frontend	http://127.0.0.1:8000/ui	Type a sentence and get sentiment
Base Endpoint	http://127.0.0.1:8000/	Welcome + usage instructions

📑 Notes
The model must be saved as: sentiment_lstm_model.keras

Ensure tokenizer name matches: "bert-base-uncased"

Swagger UI auto-generates docs from your FastAPI routes

CORS middleware is enabled to allow frontend API access


