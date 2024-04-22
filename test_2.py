import pickle
import streamlit as st
import pandas as pd
import requests
import csv
import ast
import webbrowser
# Function to fetch movie poster
def first_element_knn(movies_knn):
    first_elements = []
    with open("knn.csv", 'r') as file:
        for line in file:
            first_element = int(line.strip().split(',')[0])
            first_elements.append(first_element)
    return first_elements

def get_tmdbId_and_name(id):
    row = movies_df[movies_df['movieId'] == id]
    title = row['Title'].values[0]
    tmdb_id =row['tmdbId'].values[0]
    return tmdb_id, title

def fetch_poster(tmdbId):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=a3437eb04b986ef8745a7379ad8010d5&language=en-US".format(tmdbId)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    if poster_path != None:
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path

# Function to recommend movies
def recommend(movie, similarity, movies):
    index = movies[movies['Title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    tmdbId_list = []
    flag = 0
    for i in distances[1:6]:
        # fetch the movie poster
        tmdbId = movies.iloc[i[0]].tmdbId
        poster = fetch_poster(tmdbId)
        if poster !=  None :
            flag = flag + 1
            recommended_movie_posters.append(fetch_poster(tmdbId))
            recommended_movie_names.append(movies.iloc[i[0]].Title)
            tmdbId_list.append(tmdbId)
    return recommended_movie_names, recommended_movie_posters, flag, tmdbId_list

def recommend_b(movies_knn, movie,files):
    movieId = movies_df[movies_df['Title'] == movie].movieId.iloc[0]
    with open(files, 'r') as file:
        for line in file:
            elements = [int(x) for x in line.strip().split(',')]
            if elements[0] == movieId:
                ids = elements[1:]
    recommended_movie_names = []
    recommended_movie_posters = []
    tmdbId_list = []
    flag = 0
    for i in ids:
        # fetch the movie poster
        tmdbId,title = get_tmdbId_and_name(i)
        poster = fetch_poster(tmdbId)
        if poster !=  None :
            flag = flag + 1
            recommended_movie_posters.append(fetch_poster(tmdbId))
            recommended_movie_names.append(title)
            tmdbId_list.append(tmdbId)

    return recommended_movie_names, recommended_movie_posters, tmdbId_list,flag 

def recommend_model(movies_knn, option,files):
    with open(files, 'r') as file:
        for line in file:
            elements = [int(x) for x in line.strip().split(',')]
            if elements[0] == option:
                ids = elements[1:]
    recommended_movie_names = []
    recommended_movie_posters = []
    tmdbId_list = []
    for i in ids:
        # fetch the movie poster
        tmdbId,title = get_tmdbId_and_name(i)
        poster = fetch_poster(tmdbId)
        if poster !=  None :
            recommended_movie_posters.append(fetch_poster(tmdbId))
            recommended_movie_names.append(title)
            tmdbId_list.append(tmdbId)

    return recommended_movie_names, recommended_movie_posters, tmdbId_list

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
movies_df = pd.read_csv("data_prml.csv")
movies_knn = pd.read_csv("knn.csv")
# Streamlit UI
st.set_page_config(layout="wide")

st.title('Movie Recommender System')

# Button for selecting model
st.subheader("Content based filter")
model_selected = st.radio("Select Model:", ["Model 1(BAG OF WORDS)","Model 2(NAIVE BAYES)", "Model 3(SVM)", "Model 4(RANDOM FOREST)", "Model 5(LINEAR REGRESSION)"],index=None)
# Button for showing recommendation
if model_selected == "Model 1(BAG OF WORDS)":
    option = st.selectbox("Type or select a movie from the dropdown", movies['Title'].values)
    if st.button('Show Recommendation'):
        with st.spinner('Fetching Recommendations...'):
            recommended_movie_names, recommended_movie_posters, tmdbId_list,flag = recommend_b(movies_knn, option, "reco.csv")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i in range(min(5,flag )):
                with eval(f"col{i+1}"):
                    image_html = f'<a href="{"https://www.themoviedb.org/movie/"+str(tmdbId_list[i])}" target="_blank"><img src="{recommended_movie_posters[i]}" style="max-width:250px;"></a>'
                    st.markdown(image_html, unsafe_allow_html=True)
                    st.text(recommended_movie_names[i])
        
    if st.button('Generate Report'):
        html_file_path = "https://fazil6126912.github.io/Bag/"

    # Open HTML file in the default web browser
        webbrowser.open(html_file_path)

if model_selected == "Model 2(NAIVE BAYES)":
    option = st.selectbox("Type or select an user ID from the dropdown",first_element_knn(movies_knn))
    if st.button(f'Show Recommendation ({model_selected})'):
        with st.spinner('Fetching Recommendations...'):
            recommended_movie_names, recommended_movie_posters,tmdbId_list = recommend_model(movies_knn, option,"nb.csv")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i in range(5):
                with eval(f"col{i+1}"):
                    image_html = f'<a href="{"https://www.themoviedb.org/movie/"+str(tmdbId_list[i])}" target="_blank"><img src="{recommended_movie_posters[i]}" style="max-width:250px;"></a>'
                    st.markdown(image_html, unsafe_allow_html=True)
                    st.text(recommended_movie_names[i])

    if st.button(f'Generate Report ({model_selected})'):
        html_file_path = "https://fazil6126912.github.io/NB/index.html"

    # Open HTML file in the default web browser
        webbrowser.open(html_file_path)

if model_selected == "Model 3(SVM)":
    option = st.selectbox("Type or select an user ID from the dropdown",first_element_knn(movies_knn))
    if st.button(f'Show Recommendation ({model_selected})'):
        with st.spinner('Fetching Recommendations...'):
            recommended_movie_names, recommended_movie_posters,tmdbId_list = recommend_model(movies_knn, option,"svm.csv")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i in range(5):
                with eval(f"col{i+1}"):
                    image_html = f'<a href="{"https://www.themoviedb.org/movie/"+str(tmdbId_list[i])}" target="_blank"><img src="{recommended_movie_posters[i]}" style="max-width:250px;"></a>'
                    st.markdown(image_html, unsafe_allow_html=True)
                    st.text(recommended_movie_names[i])

    if st.button(f'Generate Report ({model_selected})'):
        html_file_path = "https://fazil6126912.github.io/SVM/"

    # Open HTML file in the default web browser
        webbrowser.open(html_file_path)

if model_selected == "Model 4(RANDOM FOREST)":
    option = st.selectbox("Type or select an user ID from the dropdown",first_element_knn(movies_knn))
    if st.button(f'Show Recommendation ({model_selected})'):
        with st.spinner('Fetching Recommendations...'):
            recommended_movie_names, recommended_movie_posters,tmdbId_list = recommend_model(movies_knn, option,"svm.csv")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i in range(5):
                with eval(f"col{i+1}"):
                    image_html = f'<a href="{"https://www.themoviedb.org/movie/"+str(tmdbId_list[i])}" target="_blank"><img src="{recommended_movie_posters[i]}" style="max-width:250px;"></a>'
                    st.markdown(image_html, unsafe_allow_html=True)
                    st.text(recommended_movie_names[i])

    if st.button(f'Generate Report ({model_selected})'):
        html_file_path = "https://fazil6126912.github.io/RF/"

    # Open HTML file in the default web browser
        webbrowser.open(html_file_path)

if model_selected == "Model 5(LINEAR REGRESSION)":
    option = st.selectbox("Type or select an user ID from the dropdown",first_element_knn(movies_knn))
    if st.button(f'Show Recommendation ({model_selected})'):
        with st.spinner('Fetching Recommendations...'):
            recommended_movie_names, recommended_movie_posters,tmdbId_list = recommend_model(movies_knn, option,"lr.csv")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i in range(5):
                with eval(f"col{i+1}"):
                    image_html = f'<a href="{"https://www.themoviedb.org/movie/"+str(tmdbId_list[i])}" target="_blank"><img src="{recommended_movie_posters[i]}" style="max-width:250px;"></a>'
                    st.markdown(image_html, unsafe_allow_html=True)
                    st.text(recommended_movie_names[i])

    if st.button(f'Generate Report ({model_selected})'):
        html_file_path = "https://fazil6126912.github.io/Lin/"

    # Open HTML file in the default web browser
        webbrowser.open(html_file_path)

if model_selected == None:
    pass

st.subheader("Item based filter")
model_selected1 = st.radio("Select Model:", ["Model 6(KNN)"],index=None)

if model_selected1 == "Model 6(KNN)":
    option = st.selectbox("Type or select an user id from the dropdown",first_element_knn(movies_knn))
    if st.button(f'Show Recommendation(KNN) ({model_selected1})'):
        with st.spinner('Fetching Recommendations...'):
            recommended_movie_names, recommended_movie_posters,tmdbId_list = recommend_model(movies_knn, option,"knn.csv")
            col1, col2, col3, col4, col5 = st.columns(5)
            for i in range(5):
                with eval(f"col{i+1}"):
                    # st.image(recommended_movie_posters[i])
                    image_html = f'<a href="{"https://www.themoviedb.org/movie/"+str(tmdbId_list[i])}" target="_blank"><img src="{recommended_movie_posters[i]}" style="max-width:250px;"></a>'
                    st.markdown(image_html, unsafe_allow_html=True)
                    st.text(recommended_movie_names[i])
    if st.button(f'Generate Report ({model_selected1})'):
        html_file_path = "https://fazil6126912.github.io/KNN/"

        webbrowser.open(html_file_path)
