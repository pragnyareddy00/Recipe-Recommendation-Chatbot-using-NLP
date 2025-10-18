# --- START: Save this code as app.py ---

import streamlit as st
import pandas as pd
import pickle
import re
import random
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1. LOAD YOUR SAVED FILES ---
try:
    recipes_df = pd.read_csv('cleaned_recipes.csv')
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 Model files not found! Ensure '.csv' and '.pkl' files are present.")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- Check for Image Files ---
cookbook_image_path = "cookbook.png"
girl_image_path = "girl.png"
cookbook_exists = os.path.exists(cookbook_image_path)
girl_exists = os.path.exists(girl_image_path)

# --- 2. RE-CREATE HELPER FUNCTIONS ---
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower(); text = re.sub(r'[^a-z\s]', '', text)
    words = text.split(); stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]; return ' '.join(words)

def recommend_recipes_enhanced(user_input, top_n=5):
    user_input_cleaned = clean_text(user_input)
    user_tfidf = vectorizer.transform([user_input_cleaned])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_scores = cosine_similarities[top_indices]
    recommendations = recipes_df.iloc[top_indices][['RecipeName', 'Ingredients', 'Cuisine', 'Steps']].copy()
    recommendations['Similarity_Score'] = top_scores
    return recommendations

# --- 3. STYLING (CSS) ---
st.set_page_config(page_title="Indian Recipe Recommender", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; padding: 1rem; }
    [data-testid="stSidebar"] * { color: #FAFAFA !important; font-size: 1.1rem !important; }
    [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stSelectbox label { font-weight: bold; }
    [data-testid="stSidebar"] .stInfo * { color: #1a1a1a !important; font-size: 1rem; }
    [data-testid="stSidebar"] .stButton button {
        background-color: #4B0082; color: #FAFAFA !important; border: none; border-radius: 8px;
        padding: 8px 15px; width: 100%; text-align: left; font-weight: bold; font-size: 1rem !important;
        margin-bottom: 8px; transition: background-color 0.2s; display: block; box-sizing: border-box;
    }
    [data-testid="stSidebar"] .stButton button:hover { background-color: #6A0DAD; }
    .main .block-container { padding: 2rem; }
    h1 { font-size: 3.2rem; font-weight: bold; color: #333; margin-bottom: 0.5rem;}
    .subtitle { font-size: 1.2rem; color: #555; margin-bottom: 2rem;}
    h2 { font-size: 2.2rem; }
    h3 { font-size: 1.7rem; font-weight: bold; color: #444;}
    .stTextInput input { font-size: 1.1rem; border-radius: 20px; border: 2px solid #555; padding: 10px 15px;}
    .stFormSubmitButton button {
        background-color: #FFD700; color: #333333 !important; border: none; border-radius: 20px;
        padding: 10px 20px; font-weight: bold; font-size: 1rem !important; transition: background-color 0.3s;
    }
    .stFormSubmitButton button:hover { background-color: #FFC700; }
    .recipe-card { background-color: #f9f9f9; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #e0e0e0;}
    .recipe-card h3 { margin-top: 0; margin-bottom: 0.5rem; }
    .recipe-card p { margin-bottom: 0.3rem; font-size: 1rem; }
    .stExpander { border: none !important; background-color: transparent !important; margin-top: 0.5rem;}
    .stExpander div[role="button"] { padding: 0.2rem 0 !important; }

    /* --- Image Positioning --- */
    .image-column-container {
        position: relative;
        height: 85vh;
        min-height: 400px;
    }
    .cookbook-img {
        position: absolute;
        top: 0;
        right: 0;
        width: 350px;
        margin-top: 1rem;
    }
    .girl-img {
        position: absolute;
        bottom: 1rem;
        right: 0;
        width: 300px;
    }
    @media (max-width: 992px) {
        .girl-img { display: none; }
        .cookbook-img {
            position: relative;
            width: 250px;
            margin: 1rem auto;
            display: block;
        }
        .image-column-container { height: auto; min-height: 200px;}
    }
    @media (max-width: 768px) {
        .cookbook-img, .girl-img, .image-column-container { display: none; }
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'recommendations' not in st.session_state: st.session_state.recommendations = pd.DataFrame()
if 'header' not in st.session_state: st.session_state.header = ""
if 'show_results_type' not in st.session_state: st.session_state.show_results_type = None
if 'quick_recipes' not in st.session_state:
    st.session_state.quick_recipes = recipes_df.sample(min(10, len(recipes_df)))

# --- SIDEBAR ---
with st.sidebar:
    st.header("Filters")
    num_results = st.slider("Number of results:", 1, 10, 5)
    st.header("Quick Recipes")
    for index, row in st.session_state.quick_recipes.iterrows():
        if st.button(row['RecipeName'], key=f"quick_{index}"):
            st.session_state.recommendations = pd.DataFrame([row])
            st.session_state.header = f"Quick Suggestion: {row['RecipeName']}"
            st.session_state.show_results_type = 'quick'
            st.rerun()


# --- MAIN LAYOUT ---
st.title("Indian Recipe Recommender")
st.markdown("<p class='subtitle'>Enter ingredients or a dish name, and I'll find the best match!</p>", unsafe_allow_html=True)

main_col, img_col = st.columns([2, 1])

with main_col:
    with st.form(key="recipe_form"):
        user_input = st.text_input("Search for recipes", placeholder="🔍 Eg: Butter Chicken or tomato, onion")
        b_col1, b_col2, _ = st.columns([1, 1, 2])
        get_recipes_button = b_col1.form_submit_button("Get Recipes!")
        surprise_me_button = b_col2.form_submit_button("🌟 Surprise Me!")

        if get_recipes_button:
            if user_input:
                recs = recommend_recipes_enhanced(user_input, top_n=num_results)
                st.session_state.recommendations = recs
                st.session_state.header = "Here are your recommendations:" if not recs.empty else "No matching recipes found."
                st.session_state.show_results_type = 'search'
            else:
                st.warning("Please enter ingredients or a dish name.")
                st.session_state.show_results_type = None
            st.rerun()


        if surprise_me_button:
            random_recipe = recipes_df.sample(1)
            st.session_state.recommendations = random_recipe
            st.session_state.header = "Surprise Recipe! 🎉"
            st.session_state.show_results_type = 'surprise'
            st.rerun()


with img_col:
    if cookbook_exists:
        st.image(cookbook_image_path, caption="", width=350)
    else:
        st.warning("cookbook.png not found in the app folder.")
    if girl_exists:
        st.image(girl_image_path, caption="", width=300)

# --- RESULTS ---
with main_col:
    if st.session_state.show_results_type is not None:
        recommendations_to_show = st.session_state.recommendations
        st.header(st.session_state.header)

        if not recommendations_to_show.empty:
            for _, row in recommendations_to_show.iterrows():
                with st.container():
                    st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                    st.markdown(f"### 🍲 {row['RecipeName']}")
                    score_display = ""
                    if st.session_state.show_results_type == 'search' and 'Similarity_Score' in row and pd.notna(row['Similarity_Score']):
                        score_display = f" | **Match Score:** {row['Similarity_Score']:.2f}"
                    st.markdown(f"<p><b>Cuisine:</b> {row['Cuisine']}{score_display}</p>", unsafe_allow_html=True)

                    with st.expander("Show Ingredients"):
                        try:
                            ingredients_list = eval(row['Ingredients']) if isinstance(row['Ingredients'], str) and row['Ingredients'].strip().startswith('[') else row['Ingredients']
                            if isinstance(ingredients_list, list): [st.markdown(f"- {item}") for item in ingredients_list]
                            else: st.write(ingredients_list)
                        except: st.write(f"Ingredients data: {row['Ingredients']}")

                    with st.expander("Show Steps"):
                        try:
                            steps_list = eval(str(row['Steps'])) if isinstance(row['Steps'], str) and str(row['Steps']).strip().startswith('[') else row['Steps']
                            if isinstance(steps_list, list): [st.markdown(f"{i}. {step}") for i, step in enumerate(steps_list, 1)]
                            else: st.write(steps_list)
                        except: st.write(f"Steps data: {row['Steps']}")

                    st.markdown('</div>', unsafe_allow_html=True)
        elif st.session_state.show_results_type == 'search':
            st.warning("😕 No recipes found matching your search. Please try different terms.")

# --- END ---
