Sentiment Analysis Using BERT

This project performs sentiment classification (Positive or Negative)** using the BERT transformer model.
The model is fine-tuned in Google Colab using HuggingFace Transformers and PyTorch.


---

Overview

This project includes the following steps:

Loading the dataset

Preprocessing text

Tokenization using BERT

Fine-tuning the BERT model

Evaluating accuracy

Making sentiment predictions

Saving the trained model



---

Technologies Used

Python

Google Colab

PyTorch

HuggingFace Transformers

Datasets library



---

Model Used

bert-base-uncased

12 Transformer layers

110M parameters

Hidden size: 768



---

Example Prediction Code

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    label = outputs.logits.argmax().item()
    return "Positive" if label == 1 else "Negative"

Example Output

predict("I love this movie!") → Positive


---

Future Improvements

Deploy model using Flask or FastAPI

Build UI with Streamlit or Gradio

Add support for multi-class emotion detection

Compare performance with models like RoBERTa or DistilBERT



---

Author

Varshini Chilukuri
NLP Project – 2025
