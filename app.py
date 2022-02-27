
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from model import predict_product



app = Flask('__name__')

@app.route('/', methods = ['POST', "GET"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    else:
        return predict_product(request.form['username'])


if __name__ == '__main__':
    app.run(debug=True)