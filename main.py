import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

df = pd.read_csv('data.csv')


def preprocess_data(df):
    if df.isnull().values.any():
        print("Дані містять пропущені значення. Пропускаємо недоступні дані.")
        df = df.dropna()
    return df


df = preprocess_data(df)

required_columns = ['Name', 'State', 'Type', 'BestTimeToVisit']
if not all(col in df.columns for col in required_columns):
    raise ValueError("Датасет не містить усіх необхідних колонок.")

tfidf = TfidfVectorizer(stop_words='english')

df['Description'] = df['Name'] + ' ' + df['State'] + ' ' + df['Type'] + ' ' + df['BestTimeToVisit']
tfidf_matrix = tfidf.fit_transform(df['Description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(destination_name, cosine_sim=cosine_sim):
    if destination_name not in df['Name'].values:
        raise ValueError(f"Напрямок '{destination_name}' не знайдено в даних.")

    idx = df.index[df['Name'] == destination_name][0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    destination_indices = [i[0] for i in sim_scores]

    return df.iloc[destination_indices]

try:
    recommended_destinations = get_recommendations('Goa Beaches')
    print("Рекомендовані напрямки:\n", recommended_destinations[['Name', 'State', 'Popularity']])
except ValueError as e:
    print(e)

def ndcg_score(recommended, relevant):
    dcg = 0.0
    for i, dest in enumerate(recommended):
        if dest in relevant:
            dcg += 1 / np.log2(i + 2)
    return dcg


relevant_destinations = ['Goa Beaches', 'Jaipur City']
recommended_list = recommended_destinations['Name'].tolist()

ndcg = ndcg_score(recommended_list, relevant_destinations)
print(f"NDCG Score: {ndcg:.4f}")