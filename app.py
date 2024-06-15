from flask import Flask, render_template, url_for, request, redirect
# import predict as pr
import bertModelPrediction as bmp
from wordcloud import WordCloud
import sklearn
app = Flask(__name__)

# Get the main keywords
def scrap(text):
  wordcloud = WordCloud().generate(text)
  words = wordcloud.words_
  keys = list(words.keys())
  result = ' '.join(keys)
  return result

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
  dic = {}
  if request.method == 'POST':
    user_text = request.form['newsHeadline']
    if user_text:
      query =  scrap(user_text)    
      # res = pr.predict(user_text)
      res = bmp.predict(user_text)
      dic = {"text":user_text, "model_res":res, "search_query":query}
      return render_template('result.html', dic=dic)
  return render_template('prediction.html')

@app.route('/search_news')
def search_news():
  return render_template('search_news.html')

@app.route('/about')
def about():
  return render_template('about.html')

if __name__=="__main__":
  app.run(debug=True)
  