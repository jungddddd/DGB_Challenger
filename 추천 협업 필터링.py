import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore

df = pd.read_csv("output_dummy.csv")

exclude_cols = ["NUM", "금융상품", "나이대"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

df_z = df.copy()
df_z[feature_cols] = df_z[feature_cols].apply(zscore)

base_categories = ['[부동산]', '[주식]', '[재테크]', '[대출]']
age_category_map = {
    10: '[교육]',
    20: '[쇼핑]',
    30: '[여행]',
    40: '[교육]',
    50: '[연금]',
    60: '[연금]'
}

def get_interest_columns(age):
    selected = base_categories + [age_category_map.get(age, '[교육]')]
    return [col for col in feature_cols if any(cat in col for cat in selected)]

def get_similar_users(target_idx, top_n=3):
    age = df.loc[target_idx, "나이대"]
    interest_cols = get_interest_columns(age)

    target_vector = df_z.loc[target_idx, interest_cols].values.reshape(1, -1)
    others = df_z[interest_cols]
    similarities = cosine_similarity(target_vector, others)[0]

    sim_df = pd.DataFrame({
        "사용자": df["NUM"],
        "유사도": similarities,
        "금융상품": df["금융상품"]
    })

    target_num = df.loc[target_idx, "NUM"]
    sim_df = sim_df[sim_df["사용자"] != target_num]
    sim_df = sim_df.sort_values(by="유사도", ascending=False).head(top_n)

    return sim_df

def recommend_products(target_idx, top_n=3):
    user_num = df.loc[target_idx, "NUM"]
    similar_users = get_similar_users(target_idx, top_n=top_n)

    product_scores = similar_users.groupby("금융상품")["유사도"].sum().sort_values(ascending=False)

    print(f"\n{user_num}번 사용자에게 추천할 금융상품:")
    for rank, (product, score) in enumerate(product_scores.items(), start=1):
        count = (similar_users["금융상품"] == product).sum()
        print(f"{rank}. {product} (추천 근거 사용자 수: {count}, 유사도 합: {round(score, 4)})")

    print(f"\n유사한 사용자 Top {top_n}:")
    print(similar_users[["사용자", "유사도", "금융상품"]].to_string(index=False))

recommend_products(target_idx=0,top_n=20)