import pandas as pd
import matplotlib.pyplot as plt
from utils import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取电影数据 + genre
genre_labels = [
    "Unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1",
                     header=None, usecols=[0, 1] + list(range(5, 24)))
movies.columns = ["item", "title"] + genre_labels

# 构建 genre_map
genre_map = dict(
    zip(
        movies["title"],
        movies[genre_labels].apply(lambda row: set(g for g in genre_labels if row[g] == 1), axis=1)
    )
)

# 读取评分数据
ratings = pd.read_csv("data/u.data", sep="\t", names=["user", "item", "rating", "timestamp"])
item_popularity = ratings["item"].value_counts().to_dict()

# 设置用户
user_id = 1
user_likes = movies[movies["item"].isin(
    ratings[(ratings["user"] == user_id) & (ratings["rating"] >= 4)]["item"]
)]["title"].tolist()
user_true = movies[movies["item"].isin(
    ratings[ratings["user"] == user_id]["item"]
)]["title"].tolist()

# 构建用户画像
user_profile_text = build_user_profile(user_id, ratings, movies, genre_labels)

# 推荐策略一：热门
candidates_pop = recommend_top_n_popular(user_id, ratings, movies, top_n=5)
llm_scores_pop = extract_scores(chatglm(get_llm_judge_prompt(user_likes, candidates_pop, user_profile_text)))
trad_scores_pop = compute_traditional_scores(user_true, candidates_pop, genre_map, item_popularity)

# 推荐策略二：协同过滤
candidates_cf = recommend_top_n_cf(user_id, ratings, movies, top_n=5)
llm_scores_cf = extract_scores(chatglm(get_llm_judge_prompt(user_likes, candidates_cf, user_profile_text)))
trad_scores_cf = compute_traditional_scores(user_true, candidates_cf, genre_map, item_popularity)

# 输出结果
print("热门推荐：", candidates_pop)
print("LLM评分（热门）：", llm_scores_pop)
print("传统评分（热门）：", trad_scores_pop)
print("协同过滤推荐：", candidates_cf)
print("LLM评分（协同过滤）：", llm_scores_cf)
print("传统评分（协同过滤）：", trad_scores_cf)

# 可视化
def plot_two_strategy_comparison(llm1, trad1, llm2, trad2, labels=["热门推荐", "协同过滤"]):
    all_keys = list(llm1.keys())
    x = range(len(all_keys))
    width = 0.2
    plt.bar([i - width*1.5 for i in x], [llm1[k] for k in all_keys], width=width, label=f"{labels[0]} - LLM")
    plt.bar([i - width*0.5 for i in x], [trad1[k] for k in all_keys], width=width, label=f"{labels[0]} - 传统")
    plt.bar([i + width*0.5 for i in x], [llm2[k] for k in all_keys], width=width, label=f"{labels[1]} - LLM")
    plt.bar([i + width*1.5 for i in x], [trad2[k] for k in all_keys], width=width, label=f"{labels[1]} - 传统")
    plt.xticks(x, all_keys)
    plt.ylim(0, 5)
    plt.title("LLM vs. 传统推荐指标对比")
    plt.ylabel("评分（0~5）")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_two_strategy_comparison(llm_scores_pop, trad_scores_pop, llm_scores_cf, trad_scores_cf)
