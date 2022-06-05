# Importing Necessary Libraries
from posixpath import split
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from flask import Flask, render_template, request

webapp=Flask(__name__)


@webapp.route('/')
def index():
    return render_template('index.html')

@webapp.route('/about')
def about():
    return render_template('about.html')

@webapp.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@webapp.route('/view')
def view():
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())



def preprocess_data(df):
    
    # Convert text to lowercase
    df['Message'] = df['Message'].str.strip().str.lower()
    return df


@webapp.route('/preprocess',methods=['POST','GET'])
def preprocess():
    global x,y,x_train, x_test, y_train, y_test,x_test,X_transformed,X_test_transformed,vec,df1,df2
    if request.method=="POST":
        size=int(request.form['split'])
        size=size/100

        
        df = pd.read_csv("spam (1).csv", encoding='latin-1')
        df = preprocess_data(df)

        # Split into training and testing data
        x = df['Message']
        y = df['Category']
        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=split, random_state=42)
        print(x)
        print(y)

        # Vectorize text reviews to numbers
        vec = CountVectorizer(stop_words='english')
        x = vec.fit_transform(x).toarray()
        x_test = vec.transform(x_test).toarray()

        print(x_test)

        return render_template('preprocess.html',msg='Data Preprocessed and Trained Successfully')
    return render_template('preprocess.html')

@webapp.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            multinomialnb = MultinomialNB()
            multinomialnb.fit(x_train,y_train)
            # Predicting the Test set results
            acc_rf = multinomialnb.score(x_test, y_test)*100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Naive Bayes Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            linearsvc = LinearSVC()
            linearsvc.fit(x_train,y_train)
            acc_dt = linearsvc.score(x_test, y_test)*100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

@webapp.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        # countvectorizer =CountVectorizer()
        multinomialnb = MultinomialNB()
        multinomialnb.fit(x_train,y_train)
        from sklearn.feature_extraction.text import CountVectorizer
        countvectorizer =CountVectorizer()
        result =multinomialnb.predict(countvectorizer.transform([f1]))
        if result==0:
            msg = 'This is a Ham Message'
        else:
            msg= 'This is a Spam Message'
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')

@webapp.route('/news')
def news():
    return render_template('news.html')



if __name__=='__main__':
    webapp.run(debug=True)