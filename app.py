import streamlit as st
import pandas as pd
import pickle as pkl
from _key import my_key
from newsapi import NewsApiClient
#@st.cache
def button_click(cv,selected_news):
	test=cv.transform(selected_news)
	fake_prob = MultiNomial.predict_proba(test)[0][1]*100
	real_prob = MultiNomial.predict_proba(test)[0][0]*100
	st.write('Probability of the given title being a fake news is:')
	st.write(fake_prob)
	st.write('Probability of the given title being a real news is:')
	st.write(real_prob)

##Getting the news headings
news_api = NewsApiClient(api_key = my_key)
def get_news(news_api):
	data = news_api.get_top_headlines(language= 'en', page_size = 20, country = 'us')
	titles = data['articles']

	heading = []
	for i in titles:
		heading.append(i['title'])
	return heading
heading = get_news(news_api)
heading = pd.Series(heading)	

####PICKLING
pickle_in = open('MultiNomial.pickle', 'rb')
MultiNomial = pkl.load(pickle_in)
pickle_in.close()

pickle_in = open('CV.pickle','rb')
cv = pkl.load(pickle_in)
pickle_in.close()

st.title('FAKE NEWS CLASSIFIER')
st.header("The machine learning model's information:")
st.write(MultiNomial)
	 
data = pd.read_csv('train.csv')
data_noid = data.drop(['id','label'], axis = 1)
st.write("<h3>First 100 rows of our data: </h3>", unsafe_allow_html = True)
st.write(data_noid.head(100))
st.write("<h4>Enter the required information </h4>", unsafe_allow_html = True)

selected_news = []
selected_news = [st.selectbox('Select the news you want to check',heading)]
if st.button('Submit'):
	button_click(cv , selected_news)


