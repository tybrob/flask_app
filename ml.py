import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
vectorizer = TfidfVectorizer()
model_nb = MultinomialNB(alpha=0.1,fit_prior=True)
model_svm = LinearSVC(C=0.3)
model_lr = LogisticRegression()

def data_preprocessing():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    submit = pd.read_csv('data/submit.csv')
    test = pd.merge(test,submit,on='id')
    total = pd.concat([train,test])

    total = total.fillna('')
    total = total.loc[total['text'] != '']

    total['text'] = total['text'].apply(stemming)
    X = total['text'].values
    Y = total['label'].values
    X = vectorizer.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, stratify=Y, random_state=42)
    return X_train,X_test,Y_train,Y_test

def stemming(content):
    port_stem = PorterStemmer()
    stop_words = stopwords.words('english')

    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def nb_train_and_evaluate(X_train,X_test,Y_train,Y_test):
    accuracy = train_and_evaluate(X_train,X_test,Y_train,Y_test,model_nb)
    return accuracy

def svm_train_and_evaluate(X_train,X_test,Y_train,Y_test):
    accuracy = train_and_evaluate(X_train,X_test,Y_train,Y_test,model_svm)
    return accuracy

def lr_train_and_evaluate(X_train,X_test,Y_train,Y_test):
    accuracy = train_and_evaluate(X_train,X_test,Y_train,Y_test,model_lr)
    return accuracy

def train_and_evaluate(X_train,X_test,Y_train,Y_test,model):
    model.fit(X_train,Y_train)
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    return str(round(test_data_accuracy,3))


def nb_predict(text_for_classification):
    prediction = predict(text_for_classification,model_nb)
    return prediction

def svm_predict(text_for_classification):
    prediction = predict(text_for_classification,model_svm)
    return prediction

def lr_predict(text_for_classification):
    prediction = predict(text_for_classification,model_lr)
    return prediction

def predict(text_for_classification,model):
    text_for_classification = stemming(text_for_classification[0])
    text_for_classification = vectorizer.transform([text_for_classification])
    prediction = model.predict(text_for_classification)
    return prediction[0]