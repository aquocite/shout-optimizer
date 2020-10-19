from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import nltk
import random
import string
import re

from sklearn.linear_model import LogisticRegression



import functools
import difflib
from ast import literal_eval

app = Flask(__name__)

clusters = pd.read_csv("huutoClusters.csv", converters = {"cTitle": literal_eval, "links": literal_eval})
clusters = clusters.drop(index = 4)

###Search algo

corpusW = []
for row in clusters['cTitle']:
    for word in row:
        if word not in corpusW:
            corpusW.append(word)
            
clusters2 = clusters.copy()
for item in corpusW:
    clusters2[item] = 0

clusters2d = clusters2.drop(['iTitleList','cNum', 'cTitle', 'mean', 'links', 'median', 'itemCount', 'priceList'], axis=1)
for index, row in clusters2.iterrows():
    for word in row['cTitle']:
        clusters2d.at[index, word] = 1

target = clusters2['cNum']
logreg = LogisticRegression()
logreg.fit(clusters2d, target)

def searchWordVectoriser(searchword):
    searchword = searchword.split()
    corpusW = []
    wordFound = False

    for row in clusters2['cTitle']:
        for word in row:
            if word not in corpusW:
                corpusW.append(word)
                
    clusters22 = clusters2.copy()
    for item in corpusW:
        clusters22[item] = 0
        
    clusters22d = clusters22.drop(['iTitleList','cNum', 'cTitle', 'mean', 'links', 'median', 'itemCount', 'priceList'], axis=1)
    clusters22d = clusters22d.head(1)
    
    
    for index, row in clusters22d.iterrows():
        for word in searchword:
            if word in clusters22d.columns:
                clusters22d.at[index, word] = 1
                wordFound = True    
    return clusters22d, wordFound

def searchFunction(gameName):
    search, wordFound = searchWordVectoriser(gameName)
    if (not wordFound):
        print('Game not found')
        return ('', False)
    else:
        pred = logreg.predict(search)[0]
        return (clusters.loc[pred], True)

#####################

@app.route('/demo/', methods=['post', 'get'])

def demo():

    item = ''
    message = ''
    f = 'Nothing found'
    mean = -9999
    median = -9999
    minimi = -1
    maksimi = -1
    titles = ''
    links = ''

    if request.method == 'POST':
        item = request.form.get('itemname') 

        if item == '':
            message = "Enter a keyword or item name"
        else:
            message = ''

        result = searchFunction(item)

        if result[1]:
            median = result[0]["median"]
            mean = result[0]["mean"]
            f = ''
            titles = str(result[0]["cTitle"])
            links = result[0]["links"]
            
            #####
            x = result[0]["priceList"]
            x = re.split('[  ]', x[1:len(x)-1:])
            x = list(filter(lambda a: a != '', x))
            floats = list(map(float, x))

            minimi = min(floats)
            maksimi = max(floats)

            ###


    return render_template('demofinal.html', item=item, titles = titles, links = links, message=message, median=median, mean=mean, f=f, minimi=minimi, maksimi=maksimi)

if __name__ == "__main__":
    app.run()