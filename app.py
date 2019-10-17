from scripts import tabledef
from scripts import forms
from scripts import helpers
from flask import Flask, redirect, url_for, render_template, request, session
import json
import sys
import os
import stripe
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd

import jieba
import jieba.analyse

import csv
import ast

app = Flask(__name__)
app.secret_key = os.urandom(12)  # Generic key for dev purposes only

stripe_keys = {
  'secret_key': os.environ['STRIPE_SECRET_KEY'],
  'publishable_key': os.environ['STRIPE_PUBLISHABLE_KEY']
}

stripe.api_key = stripe_keys['secret_key']

# Heroku
#from flask_heroku import Heroku
#heroku = Heroku(app)

# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    user.active = user.payment == helpers.payment_token()
    user.key = stripe_keys['publishable_key']
    return render_template('home.html', user=user)

# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))

# -------- Charge ---------------------------------------------------------- #
@app.route('/charge', methods=['POST'])
def charge():
    if session.get('logged_in'):
        user = helpers.get_user()
        try:
            amount = 1000   # amount in cents
            customer = stripe.Customer.create(
                email= user.email,
                source=request.form['stripeToken']
            )
            stripe.Charge.create(
                customer=customer.id,
                amount=amount,
                currency='usd',
                description='Resume Scanner Donation'
            )
            helpers.change_user(payment=helpers.payment_token())
            user.active = True
            return render_template('home.html', user=user)
        except stripe.error.StripeError:
            return render_template('error.html')

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        df = pd.read_csv(file_path)
        
        seg_list01 = df['job-description']
        seg_list02 = df['your-resume']

        item01_list = seg_list01
        item01 = ','.join(item01_list)

        item02_list = seg_list02
        item02 = ','.join(item02_list)

        documents = [item01, item02]

        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(documents)

        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['item01', 'item02'])

        answer = cosine_similarity(df, df)
        answer = pd.DataFrame(answer)
        answer = answer.iloc[[1],[0]].values[0]
        answer = round(float(answer),4)*100
        
        return "Your resume matched " + str(answer) + " %" + " of the job-description!"
    return None

# ======== Main ============================================================== #
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)