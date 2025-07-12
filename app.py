from flask import Flask, request, render_template
import re
import string
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

app = Flask(__name__)

# Load pre-trained models and vectorizer
LR = joblib.load('LR_model.pkl')
DT = joblib.load('DT_model.pkl')
GB = joblib.load('GB_model.pkl')
RF = joblib.load('RF_model.pkl')
vectorization = joblib.load('vectorizer.pkl')

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def predict_news(news):
    news_processed = [wordopt(news)]
    news_vectorized = vectorization.transform(news_processed)
    pred_LR = LR.predict(news_vectorized)[0]
    pred_DT = DT.predict(news_vectorized)[0]
    pred_GB = GB.predict(news_vectorized)[0]
    pred_RF = RF.predict(news_vectorized)[0]
    
    labels = ["Fake News", "Real News"]
    return {
        "Logistic Regression": labels[pred_LR],
        "Decision Tree": labels[pred_DT],
        "Gradient Boosting": labels[pred_GB],
        "Random Forest": labels[pred_RF]
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        news_article = request.form["news_article"]
        predictions = predict_news(news_article)
        return render_template("index.html", predictions=predictions, news_article=news_article)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
