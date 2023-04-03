import pickle
from flask import Flask,Response,Request,jsonify,make_response,json,request,render_template
import re
import tensorflow as tf
from sklearn.feature_extraction import text
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
app=Flask(__name__)

stop_words = text.ENGLISH_STOP_WORDS


@app.route('/predict',methods=['GET'])
def index():
    review = request.args.get('review')
    print(review)
    with open('tokenizer1.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle) 
    cleanedReview = clean_review(review,stop_words)
    tokenized_ip=tokenizer.texts_to_sequences([cleanedReview])
    fin=pad_sequences(tokenized_ip,padding='pre', maxlen = 500)
    model = tf.keras.models.load_model('./cloth_model.h5')
    # print("Type of clean review",type(cleanedReview))
    res = model.predict(fin)
    print(round(res[0][0]))
    if(round(res[0][0])==1):
        sent="Positive"
    else:
        sent="Negative"
    return render_template('output.html',sentiment = sent)

@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html')



def clean_review(review, stopwords):
    review=review.lower()
    html_tag = re.compile('<.*?>')
    cleaned_review = re.sub(html_tag, "", review).split()
    cleaned_review = [i for i in cleaned_review if i not in stopwords]
    return " ".join(cleaned_review)


# def clean_review2(review, stopwords):
#     # html_tag = re.compile('<.*?>')
#     # cleaned_review = re.sub(html_tag, "", review).split()
#     cleaned_review = review.split()
#     cleaned_review = [i for i in cleaned_review if i not in stopwords]
#     return " ".join(cleaned_review)


if(__name__=="__main__"):
    app.run(debug=True)