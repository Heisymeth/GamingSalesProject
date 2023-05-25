import streamlit as st
import pandas as pd
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.sidebar.subheader("Gaming Sales")
pages = st.sidebar.radio("Contents",options = ['Introduction','Visualization','Classification','Regression'])
st.sidebar.subheader("Contact")
st.sidebar.markdown("""Grégoire Apostoloff <a href="https://www.linkedin.com/in/gregoire-apostoloff/"> :link: </a>""", unsafe_allow_html=True)
st.sidebar.markdown("""Amaury Maros <font color='blue'><a href="https://www.linkedin.com/in/amaurymaros/">  :link:</a>""", unsafe_allow_html=True)
st.sidebar.markdown("""Nabil Larif""")
st.sidebar.image("datascientest.png")


with open('/home/linux/Documents/Streamlit_GamingSales/models/dtc.pkl', 'rb') as model_file:
  dtc = pickle.load(model_file)

with open('/home/linux/Documents/Streamlit_GamingSales/models/knn_clf.pkl', 'rb') as model_file:
  knn_clf = pickle.load(model_file)

with open('/home/linux/Documents/Streamlit_GamingSales/models/dtr.pkl', 'rb') as model_file:
  dtr = pickle.load(model_file)

with open('/home/linux/Documents/Streamlit_GamingSales/models/knn_reg.pkl', 'rb') as model_file:
  knn_reg = pickle.load(model_file)


df = pd.read_csv('/home/linux/Documents/Streamlit_GamingSales/csv_file.csv')

# Classification
X = pd.read_csv('/home/linux/Documents/Streamlit_GamingSales/models/X.csv')
Y_clf = pd.read_csv('/home/linux/Documents/Streamlit_GamingSales/models/clf_Y.csv')
Y_reg = pd.read_csv('/home/linux/Documents/Streamlit_GamingSales/models/reg_Y.csv')


if pages == 'Introduction':
    st.title("Gaming sales predictions")

    col1, col2, col3 = st.columns([1,6,1])
    with col1:
         st.write("")
    with col2:
         st.image('jeux-videos.jpg', width = 500)
    with col3 :
         st.write("")
    
    st.markdown("""<div style="text-align: justify;"> \
            The video game industry has become one of the most dynamic and lucrative sectors in the entertainment market,\
                  generating billions of dollars in revenue annually on a global scale. To understand the trends and performance\
                      of this constantly evolving market, it is essential to conduct in-depth analysis on the available data.\
            </div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify;"> \
    
            For our project, we focused on the vgsales dataset from <a href="https://www.kaggle.com/datasets/gregorut/videogamesales">Kaggle</a>\
                  to understand video game sales, a problem that greatly interests companies in the industry. This dataframe contains information\
                      on sales ranking, game title, release platform, release year, game genre, publisher, as well as sales in different regions of\
                          the world.\
                              
            </div>""", unsafe_allow_html=True)
    
if pages == 'Visualization':
    st.title('Visualization with PowerBI')

    st.subheader('Initial & web scraping upgraded dataframe visualization')

    # Personnalisation CSS pour agrandir l'iframe
    css = """
        <style>
        iframe {
            width: 150%;
            height: 600px;
        }
        </style>
        """

    # Affichage du CSS personnalisé
    st.markdown(css, unsafe_allow_html=True)

    # Affichage de l'iframe contenant le rapport Power BI
    st.components.v1.iframe("https://app.powerbi.com/reportEmbed?reportId=a3a4cd49-8798-4a3e-b348-1368b89c7192&autoAuth=true&embeddedDemo=true")
    # Login :   GregoireApostoloff@DataWorld606.onmicrosoft.com
    # Mdp :     #44f4#NCFpDxrLct

if pages == 'Classification':


    st.subheader("Classification prediction")
    st.markdown('Try our classification model to get a 1 to 4 mark to predict your hypothetic commercial success')
    st.subheader("Choose your modelisation's characteristics :")
    col1, col2 = st.columns(2)

    with col1:
            model = st.selectbox("Classification model", options = ['Decision Tree','KNN'])
            genre = st.selectbox("Game's genre", options = df['Genre'].value_counts().index)
            publisher = st.selectbox("Game's publisher", options = df['Publisher'].value_counts().index)
            platform = st.selectbox("Platform", options = df['Platform'].value_counts().index)
        
    with col2:
            Test_MC = st.slider('Desired MetaCritic redaction mark', 0.0, 10.0, 5.0, step = 0.5)
            Test_JV = st.slider('Desired JeuxVideos.com redaction mark', 0.0, 10.0, 5.0, step = 0.5)
            Players_MC = st.slider('Desired MetaCritic players mark', 0.0, 10.0, 5.0, step = 0.5)
            Players_JV = st.slider('Desired JeuxVideos.com players mark', 0.0, 10.0, 5.0, step = 0.5)
 
    clf_test = pd.DataFrame(0,index = [0], columns = X.columns)

    for i in X.columns:
        if (i == 'genre_'+genre) :
            clf_test[i] = 1
    for i in X.columns:
        if (i == 'publisher_'+publisher) :
            clf_test[i] = 1
    for i in X.columns:
        if (i == 'platform_'+platform) :
            clf_test[i] = 1

    clf_test['Test_MC'], clf_test['Test_JV'], clf_test['Players_MC'], clf_test['Players_JV'] = Test_MC, Test_JV, Players_MC, Players_JV

    st.subheader("Have a look on your test dataframe just below :")
    st.dataframe(clf_test)

    st.subheader("Click on the button to get your prediction")

    b = st.button("Click to see your result")
    if b:
        if model == 'Decision Tree':
            pred = dtc.predict(clf_test)
            st.write("Your commercial success is evaluate to ", str(pred[0]),"/ 4")
        if model == 'KNN':
            pred = knn_clf.predict(clf_test)
            st.write("Your commercial success is evaluate to ", str(pred[0]),"/ 4")

if pages == 'Regression':
    st.subheader("Regression prediction")
    st.markdown('Try our regression model to get an estimation of sales number')
    st.subheader("Choose your modelisation's characteristics :")
    col1, col2 = st.columns(2)

    with col1:
                model = st.selectbox("Regression model", options = ['Decision Tree Regressor', 'KNN Regressor'])
                genre = st.selectbox("Game's genre", options = df['Genre'].value_counts().index)
                publisher = st.selectbox("Game's publisher", options = df['Publisher'].value_counts().index)
                platform = st.selectbox("Platform", options = df['Platform'].value_counts().index)
            
    with col2:
                Test_MC = st.slider('Desired MetaCritic redaction mark', 0.0, 10.0, 5.0, step = 0.5)
                Test_JV = st.slider('Desired JeuxVideos.com redaction mark', 0.0, 10.0, 5.0, step = 0.5)
                Players_MC = st.slider('Desired MetaCritic players mark', 0.0, 10.0, 5.0, step = 0.5)
                Players_JV = st.slider('Desired JeuxVideos.com players mark', 0.0, 10.0, 5.0, step = 0.5)
    
    data = {'Platform':platform,'Genre':genre,'Publisher':publisher,
            'Test_JV':Test_JV,'Players_JV':Players_JV,
            'Test_MC':Test_MC,'Players_MC':Players_MC}
    
    reg_test = pd.DataFrame(0,index = [0], columns = X.columns)

    for i in X.columns:
        if (i == 'genre_'+genre) :
            reg_test[i] = 1
    for i in X.columns:
        if (i == 'publisher_'+publisher) :
            reg_test[i] = 1
    for i in X.columns:
        if (i == 'platform_'+platform) :
            reg_test[i] = 1

    reg_test['Test_MC'], reg_test['Test_JV'], reg_test['Players_MC'], reg_test['Players_JV'] = Test_MC, Test_JV, Players_MC, Players_JV

    st.subheader("Have a look on your test dataframe just below :")
    st.dataframe(reg_test)

    st.subheader("Click on the button to get your prediction")

    b = st.button("Click to see your result")
    if b:
        if model == 'Decision Tree Regressor':
            reg_pred = dtr.predict(reg_test)
            st.write("The model estimate your global sales to", float(reg_pred), "millions")
        if model == 'KNN Regressor':
            reg_pred = knn_reg.predict(reg_test)
            st.write("The model estimate your global sales to", float(reg_pred), "millions")