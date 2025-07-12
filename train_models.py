import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Load data
data_fake = pd.read_csv('task1\Fake.csv')
data_true = pd.read_csv('task1\True.csv')

# Prepare data
data_fake["class"] = 0
data_true['class'] = 1

# Use .copy() to avoid the SettingWithCopyWarning
data_fake_manual_testing = data_fake.tail(10).copy()
data_fake = data_fake.iloc[:-10]

data_true_manual_testing = data_true.tail(10).copy()
data_true = data_true.iloc[:-10]

data_fake_manual_testing.loc[:, 'class'] = 0
data_true_manual_testing.loc[:, 'class'] = 1

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)
data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train models
LR = LogisticRegression()
LR.fit(xv_train, y_train)
joblib.dump(LR, 'LR_model.pkl')

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
joblib.dump(DT, 'DT_model.pkl')

GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)
joblib.dump(GB, 'GB_model.pkl')

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)
joblib.dump(RF, 'RF_model.pkl')

joblib.dump(vectorization, 'vectorizer.pkl')

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"
    
    print(f"{model_name} Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("AUC-ROC:", auc_roc)
    print(classification_report(y_test, y_pred))
    print("-" * 50)

evaluate_model(LR, xv_test, y_test, "Logistic Regression")
evaluate_model(DT, xv_test, y_test, "Decision Tree")
evaluate_model(GB, xv_test, y_test, "Gradient Boosting")
evaluate_model(RF, xv_test, y_test, "Random Forest")