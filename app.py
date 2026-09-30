import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Movie Dataset
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Title': [
            'Inception', 'Interstellar', 'The Dark Knight', 'The Matrix', 
            'Avengers: Endgame', 'The Notebook', 'La La Land', 'Titanic',
            'The Conjuring', 'A Quiet Place'
        ],
        'Genre': [
            'Sci-Fi Action Mind-bending', 'Sci-Fi Space Drama', 'Action Crime Superhero', 'Sci-Fi Action Cyberpunk',
            'Action Superhero Sci-Fi', 'Romance Drama Love', 'Romance Music Drama', 'Romance Drama History',
            'Horror Supernatural Scary', 'Horror Sci-Fi Thriller'
        ]
    })

df = load_data()


# Compute Similarity Matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Header
st.title("🎬 Movie Recommendation App")
st.markdown("Discover movie recommendations based on genre similarity using **Cosine Similarity**.")

# Selection
selected_movie = st.selectbox("Select a movie you like:", df['Title'])

# Recommendation Logic
idx = df[df['Title'] == selected_movie].index[0]
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]

st.subheader("Top 3 Recommendations")
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for i, (movie_idx, score) in enumerate(sim_scores):
    with cols[i]:
        st.metric(
            label=df.iloc[movie_idx]['Title'], 
            value=f"{int(score * 100)}% Match",
            help=f"Genre: {df.iloc[movie_idx]['Genre']}"
        )
        st.caption(f"**Genre:** {df.iloc[movie_idx]['Genre']}")