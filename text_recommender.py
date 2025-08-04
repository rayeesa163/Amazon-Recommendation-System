import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# STEP 1: Load dataset manually
# -------------------------------
print("⏳ Loading dataset...")

labels = []
reviews = []

with open("amazon.reviews.csv", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20000:   # read only first 20k lines for speed
            break
        parts = line.strip().split(" ", 1)  # split into 2 parts only
        if len(parts) == 2:
            labels.append(parts[0])
            reviews.append(parts[1])

df = pd.DataFrame({"Label": labels, "Review": reviews})
print("✅ Dataset loaded:", df.shape)
print(df.head())

# -------------------------------
# STEP 2: Convert labels to ratings
# -------------------------------
df["Rating"] = df["Label"].apply(lambda x: 5 if "__label__2" in x else 1)

# -------------------------------
# STEP 3: Convert reviews to TF-IDF
# -------------------------------
print("⏳ Converting reviews to TF-IDF features...")

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["Review"])

print("✅ TF-IDF Matrix shape:", tfidf_matrix.shape)

# -------------------------------
# STEP 4: Compute similarity
# -------------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("✅ Cosine similarity matrix created")

# -------------------------------
# STEP 5: Recommendation function
# -------------------------------
def recommend_similar_reviews(index, top_n=5):
    if index >= len(df):
        return ["Index out of range!"]

    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip self

    recommendations = []
    for i, score in sim_scores:
        recommendations.append((df.iloc[i]["Review"], df.iloc[i]["Rating"], round(score, 3)))

    return recommendations

# -------------------------------
# STEP 6: Test
# -------------------------------
sample_index = 10
print("\n⭐ Original Review:")
print(df.iloc[sample_index]["Review"])
print("Rating:", df.iloc[sample_index]["Rating"])

print("\n🎁 Recommended Similar Reviews:")
for rec, rating, score in recommend_similar_reviews(sample_index, top_n=5):
    print(f"\n➡️ {rec}\n   (Rating: {rating}, Similarity: {score})")
