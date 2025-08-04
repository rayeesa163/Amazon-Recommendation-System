import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# -------------------------------
# ✅ Must be first Streamlit command
# -------------------------------
st.set_page_config(page_title="Amazon Recommender Clone", page_icon="🛒", layout="wide")

# -------------------------------
# STEP 1: Load dataset
# -------------------------------
@st.cache_data
def load_data(limit=5000):
    labels, reviews = [], []
    with open("amazon.reviews.csv", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels.append(parts[0])
                reviews.append(parts[1])
    df = pd.DataFrame({"Label": labels, "Review": reviews})
    df["Rating"] = df["Label"].apply(lambda x: 5 if "__label__2" in x else random.randint(2, 4))
    return df

df = load_data()

# -------------------------------
# STEP 2: Build TF-IDF model
# -------------------------------
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = tfidf.fit_transform(df["Review"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, cosine_sim

tfidf, cosine_sim = build_model(df)

# -------------------------------
# STEP 3: Recommendation function
# -------------------------------
def recommend_products(text, top_n=9):
    vec = tfidf.transform([text])
    sim_scores = cosine_similarity(vec, tfidf.transform(df["Review"])).flatten()
    sim_indices = sim_scores.argsort()[::-1][1:top_n+1]
    results = []
    for i in sim_indices:
        product = {
            "title": f"Product {i+1}",
            "desc": df.iloc[i]["Review"][:80] + "...",
            "rating": df.iloc[i]["Rating"],
            "price": f"${random.randint(50, 500)}",
            "image": f"https://picsum.photos/200?random={i}"  # placeholder product image
        }
        results.append(product)
    return results

# -------------------------------
# STEP 4: Amazon Clone UI
# -------------------------------
st.title("🛒 Amazon Product Recommender (Clone)")
st.write("Get recommendations like **Customers who bought this also bought**")

user_input = st.text_area("✍️ Describe what you like:", height=120,
                          placeholder="Example: I want a phone with good battery and camera")

if st.button("🔍 Get Recommendations"):
    if user_input.strip():
        st.subheader("📦 Recommended Products:")

        recs = recommend_products(user_input, top_n=9)
        cols = st.columns(3)  # 3 products per row

        for idx, product in enumerate(recs):
            with cols[idx % 3]:
                stars = "⭐" * product["rating"] + "☆" * (5 - product["rating"])
                st.markdown(
                    f"""
                    <div style="background:white; padding:15px; border-radius:10px;
                                margin-bottom:15px; box-shadow:0px 2px 6px rgba(0,0,0,0.15); text-align:center">
                        <img src="{product['image']}" width="150" style="border-radius:8px"><br><br>
                        <b>{product['title']}</b><br>
                        <small>{product['desc']}</small><br><br>
                        <span style="color:#ffa41c; font-size:18px;">{stars}</span><br>
                        <b style="color:green; font-size:16px;">{product['price']}</b><br><br>
                        <button style="background:#ffa41c; border:none; padding:8px 16px;
                                       border-radius:5px; cursor:pointer; font-weight:bold;">
                            🛒 Add to Cart
                        </button>
                        <button style="background:#ff9900; border:none; padding:8px 16px;
                                       border-radius:5px; cursor:pointer; font-weight:bold; margin-left:5px;">
                            ⚡ Buy Now
                        </button>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("⚠️ Please enter a description first!")
