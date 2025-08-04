import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# STEP 1: Load dataset
# -------------------------------
print("⏳ Loading dataset...")

df = pd.read_csv(
    "amazon.reviews.csv",
    sep="\t",            # dataset is tab-separated
    nrows=20000,         # use smaller chunk for testing
    on_bad_lines="skip",
    engine="python",
    encoding="utf-8",
    header=None          # this dataset has no headers
)

print("✅ Dataset loaded with shape:", df.shape)
print(df.head())

# -------------------------------
# STEP 2: Prepare dataset
# -------------------------------
# Column 0 = label, Column 1 = review text
df = df.rename(columns={0: "Label", 1: "Review"})

# Convert labels to numeric ratings
df["Rating"] = df["Label"].apply(lambda x: 5 if "__label__2" in x else 1)

# Create fake Users and Products
df["User"] = ["U" + str(i) for i in range(len(df))]
df["Product"] = ["P" + str(i) for i in range(len(df))]

print("\n✅ Cleaned dataset sample:")
print(df.head())

# -------------------------------
# STEP 3: Build User-Item Matrix
# -------------------------------
user_item_matrix = df.pivot_table(
    index="User",
    columns="Product",
    values="Rating"
).fillna(0)

print("\n✅ User-Item Matrix shape:", user_item_matrix.shape)

# -------------------------------
# STEP 4: Compute Similarity
# -------------------------------
similarity = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(
    similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

print("\n✅ User Similarity Matrix created")

# -------------------------------
# STEP 5: Recommendation Function
# -------------------------------
def recommend_products(user, user_item_matrix, similarity_df, top_n=5):
    if user not in similarity_df.index:
        return ["User not found!"]

    similar_user = similarity_df[user].sort_values(ascending=False).index[1]

    similar_user_products = user_item_matrix.loc[similar_user]
    target_user_products = user_item_matrix.loc[user]

    recommendations = similar_user_products[
        (similar_user_products > 0) & (target_user_products == 0)
    ]

    if recommendations.empty:
        return ["No new recommendations available."]

    return recommendations.index.tolist()[:top_n]

# -------------------------------
# STEP 6: Test
# -------------------------------
sample_user = user_item_matrix.index[0]
print("\n🎁 Recommendations for", sample_user, ":")
print(recommend_products(sample_user, user_item_matrix, similarity_df))

