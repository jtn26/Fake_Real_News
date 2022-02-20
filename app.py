import flask
import pickle
from newspaper import Article
import re, string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


with open(f'models/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'models/countVectorizer.pkl', 'rb') as f:
    countVec = pickle.load(f)

with open(f'models/clickbaitModel.pkl', 'rb') as f:
    cmodel = pickle.load(f)

with open(f'models/clickbaitVectorizer.pkl', 'rb') as f:
    clickVec = pickle.load(f)

with open(f'models/categoryModel.pkl', 'rb') as f:
    amodel = pickle.load(f)

with open(f'models/categoryVectorizer.pkl', 'rb') as f:
    aVec = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        url = flask.request.form['url']
        try:
            res = "Real or Fake: "
            cres = "Clickbait or Not: "
            ares = "Category of Article: "
            article = Article(url)
            article.download()
            article.parse()

            ar_title = article.title

            input_variables = article.text
            x = preprocess(input_variables)
            df_testing = pd.DataFrame([x], columns = ['val'])
            a = countVec.transform(df_testing['val'].apply(lambda x: np.str_(x)))
            
            prediction = model.predict(a)

            if prediction[0] == 0:
                res += "Fake News"
            else:
                res += "True News"

            ar_title = preprocess(ar_title)
            ar_testing = pd.DataFrame([ar_title], columns = ['val'])
            b = clickVec.transform(ar_testing['val'].apply(lambda x: np.str_(x)))
            pred = cmodel.predict(b)

            if pred[0] == 0:
                cres += 'Clickbait'
            else:
                cres += 'Not clickbait'
            
            #c = aVec.transform(df_testing['val'].apply(lambda x: np.str_(x)))
            #cc = amodel.predict(c)
            #ares += np.mode(cc)
            

            if input_variables == "":
                res = "Invalid Input. Please try again."
                cres = ""
            #    ares = ""
        except:
            res = "Invalid Input. Please try again."
            cres = ""
            ares = ""
            
        return flask.render_template('main.html',
                                        original_input={'URL':url},
                                        result1=res,
                                        result2=cres,
                                        #result3=ares
                                        )

def preprocess(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(' +',' ',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    #text = BeautifulSoup(text,  features="lxml").text 
    text = re.sub('https?://\S+|www\.\S+', ' ', text)

    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    # text = ' '.join([token.lemma_ for token in list(nlp(text)) if (token.is_stop == False)])
    return text

if __name__ == '__main__':
     app.run()