import numpy as np
import pandas as pd
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

# 热门推荐（按 item 频率）
def recommend_top_n_popular(user_id, ratings, movies, top_n=5):
    user_rated = ratings[ratings["user"] == user_id]["item"].tolist()
    item_scores = ratings[~ratings["item"].isin(user_rated)]["item"].value_counts()
    top_items = item_scores.head(top_n).index.tolist()
    return movies[movies["item"].isin(top_items)]["title"].tolist()

# 协同过滤推荐
def recommend_top_n_cf(user_id, ratings, movies, top_n=5):
    user_item_matrix = ratings.pivot(index='user', columns='item', values='rating').fillna(0)
    if user_id not in user_item_matrix.index:
        return []
    similarity_matrix = cosine_similarity(user_item_matrix)
    user_index = user_id - 1
    user_similarity = similarity_matrix[user_index]
    weighted_ratings = user_similarity @ user_item_matrix.values
    normalization = np.abs(user_similarity).sum()
    recommendation_scores = weighted_ratings / (normalization + 1e-8)
    scores_series = pd.Series(recommendation_scores, index=user_item_matrix.columns)
    rated_items = ratings[ratings['user'] == user_id]['item'].tolist()
    scores_series = scores_series.drop(labels=rated_items, errors='ignore')
    top_items = scores_series.sort_values(ascending=False).head(top_n).index.tolist()
    return movies[movies['item'].isin(top_items)]['title'].tolist()

# 构造上下文增强 Prompt
def get_llm_judge_prompt(user_likes, candidates, user_profile_text):
    return (
        f"{user_profile_text}\n"
        f"推荐列表是：{', '.join(candidates)}。\n"
        "请从以下四个维度为推荐列表打分（每项1-5）并解释理由：\n"
        "1. 相关性\n2. 多样性\n3. 新颖性\n4. 公平性"
    )

# 模拟 LLM 调用（可替换为真实API）
def chatglm(prompt):
    print("[模拟 LLM 调用] Prompt：\n", prompt)
    return "相关性: 4\n多样性: 3\n新颖性: 4\n公平性: 5"

# 提取评分
def extract_scores(text):
    pattern = r"(相关性|多样性|新颖性|公平性).*?([1-5])"
    matches = re.findall(pattern, text)
    return {k: int(v) for k, v in matches}

# 推荐评估指标
def precision_at_k(rel, k):
    return sum(rel[:k]) / k if k else 0

def ndcg_score(y_true, y_score):
    def dcg(relevances):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    idcg = dcg(sorted(y_true[0], reverse=True))
    return dcg(y_true[0]) / idcg if idcg > 0 else 0

def ild(candidates, genre_map):
    from itertools import combinations
    def genre_distance(g1, g2):
        return 1 - len(g1 & g2) / len(g1 | g2) if g1 | g2 else 0
    pairs = combinations(candidates, 2)
    distances = [genre_distance(genre_map.get(i, set()), genre_map.get(j, set())) for i, j in pairs]
    return np.mean(distances) if distances else 0

def compute_novelty(candidates, item_popularity):
    scores = []
    for title in candidates:
        pop = item_popularity.get(title, 1)
        novelty = 1 / np.log(1 + pop)
        scores.append(novelty)
    return np.mean(scores)

def compute_fairness(candidates, genre_map, user_true):
    rec_genres = set(g for title in candidates for g in genre_map.get(title, set()))
    true_genres = set(g for title in user_true for g in genre_map.get(title, set()))
    if not true_genres:
        return 0
    return len(rec_genres & true_genres) / len(true_genres)

def compute_traditional_scores(y_true, candidates, genre_map, item_popularity, k=5):
    rel = [1 if item in y_true else 0 for item in candidates[:k]]
    relevance = precision_at_k(rel, k)
    diversity = ild(candidates[:k], genre_map)
    novelty = compute_novelty(candidates[:k], item_popularity)
    fairness = compute_fairness(candidates[:k], genre_map, y_true)
    return {
        "相关性": round(float(relevance) * 5, 2),
        "多样性": round(float(diversity) * 5, 2),
        "新颖性": round(min(float(novelty) * 5, 5), 2),
        "公平性": round(float(fairness) * 5, 2)
    }

# 用户画像构造函数
def build_user_profile(user_id, ratings, movies, genre_labels):
    liked_items = ratings[(ratings["user"] == user_id) & (ratings["rating"] >= 4)]["item"].tolist()
    liked_movies = movies[movies["item"].isin(liked_items)]
    genre_counts = liked_movies[genre_labels].sum().sort_values(ascending=False)
    top_genres = genre_counts[genre_counts > 0].index.tolist()[:3]
    movie_examples = liked_movies["title"].head(3).tolist()
    desc = f"用户偏好类型：{'、'.join(top_genres)}，喜欢的电影有：{'、'.join(movie_examples)}。"
    return desc
