# ========================================
# ЛАБОРАТОРНАЯ РАБОТА 4: РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА
# Эксперимент 3: Рекомендации аниме на основе пользовательских оценок
# ========================================

# ## Импорт необходимых библиотек

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ## Загрузка данных

print("=" * 80)
print("ЭКСПЕРИМЕНТ 3: РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
print("=" * 80)
print("\n1. Загрузка данных...\n")

# """
# Рекомендательная система предсказывает, какие аниме понравятся пользователю
# на основе его предыдущих оценок и оценок похожих пользователей.
# 
# Два основных подхода:
# 1. Collaborative Filtering (коллаборативная фильтрация):
#    - User-based: находим похожих пользователей
#    - Item-based: находим похожие аниме
# 2. Content-based: рекомендации на основе характеристик аниме
# """

# Загружаем данные об аниме
anime_df = pd.read_csv('../data/processed/anime_processed.csv')
print(f"Датасет аниме: {anime_df.shape}")

# Загружаем пользовательские оценки - файл большой, читаем частями
print("\nЗагрузка пользовательских оценок (большой файл)...")
# Читаем только первые N строк для примера (полный датасет очень большой)
chunk_size = 1000000  # 1 миллион строк
ratings_df = pd.read_csv('../data/processed/users_score_processed.csv', 
                         nrows=chunk_size)
print(f"Датасет оценок: {ratings_df.shape}")
print(f"Колонки: {ratings_df.columns.tolist()}")
print(f"\nПример данных:")
print(ratings_df.head(10))

# ## Анализ данных

print("\n" + "=" * 80)
print("2. Анализ данных")
print("=" * 80 + "\n")

print(f"Количество уникальных пользователей: {ratings_df['user_id'].nunique()}")
print(f"Количество уникальных аниме: {ratings_df['anime_id'].nunique()}")
print(f"Общее количество оценок: {len(ratings_df)}")

# Статистика оценок
print(f"\nРаспределение оценок:")
print(ratings_df['rating'].value_counts().sort_index())

print(f"\nСтатистика:")
print(f"  Средняя оценка: {ratings_df['rating'].mean():.2f}")
print(f"  Медианная оценка: {ratings_df['rating'].median():.2f}")
print(f"  Стандартное отклонение: {ratings_df['rating'].std():.2f}")

# Активность пользователей
user_activity = ratings_df.groupby('user_id').size()
print(f"\nАктивность пользователей:")
print(f"  Среднее количество оценок на пользователя: {user_activity.mean():.2f}")
print(f"  Медианное количество оценок: {user_activity.median():.0f}")
print(f"  Максимальное количество оценок: {user_activity.max()}")

# Популярность аниме
anime_popularity = ratings_df.groupby('anime_id').size()
print(f"\nПопулярность аниме:")
print(f"  Среднее количество оценок на аниме: {anime_popularity.mean():.2f}")
print(f"  Медианное количество оценок: {anime_popularity.median():.0f}")
print(f"  Максимальное количество оценок: {anime_popularity.max()}")

# ## Подготовка данных для рекомендательной системы

print("\n" + "=" * 80)
print("3. Подготовка данных")
print("=" * 80 + "\n")

# Фильтруем данные: оставляем только активных пользователей и популярные аниме
# Это уменьшает разреженность матрицы и улучшает качество рекомендаций

min_user_ratings = 20  # Минимум оценок от пользователя
min_anime_ratings = 50  # Минимум оценок для аниме

print(f"Фильтрация данных:")
print(f"  Минимум оценок от пользователя: {min_user_ratings}")
print(f"  Минимум оценок для аниме: {min_anime_ratings}")

# Подсчитываем количество оценок
user_counts = ratings_df['user_id'].value_counts()
anime_counts = ratings_df['anime_id'].value_counts()

# Фильтруем
active_users = user_counts[user_counts >= min_user_ratings].index
popular_anime = anime_counts[anime_counts >= min_anime_ratings].index

ratings_filtered = ratings_df[
    (ratings_df['user_id'].isin(active_users)) & 
    (ratings_df['anime_id'].isin(popular_anime))
].copy()

print(f"\nПосле фильтрации:")
print(f"  Пользователей: {ratings_filtered['user_id'].nunique()}")
print(f"  Аниме: {ratings_filtered['anime_id'].nunique()}")
print(f"  Оценок: {len(ratings_filtered)}")

# Создаем маппинг для более компактных индексов
user_mapping = {user_id: idx for idx, user_id in enumerate(ratings_filtered['user_id'].unique())}
anime_mapping = {anime_id: idx for idx, anime_id in enumerate(ratings_filtered['anime_id'].unique())}

ratings_filtered['user_idx'] = ratings_filtered['user_id'].map(user_mapping)
ratings_filtered['anime_idx'] = ratings_filtered['anime_id'].map(anime_mapping)

# ## Разделение на train/test

print("\n" + "=" * 80)
print("4. Разделение на train/test")
print("=" * 80 + "\n")

# """
# Для рекомендательных систем важно правильно разделить данные:
# - Для каждого пользователя оставляем часть оценок для тестирования
# - Это позволяет оценить, насколько хорошо модель предсказывает оценки
# """

train_data, test_data = train_test_split(ratings_filtered, test_size=0.2, random_state=42)

print(f"Обучающая выборка: {len(train_data)} оценок")
print(f"Тестовая выборка: {len(test_data)} оценок")

# ## Модель 1: User-Based Collaborative Filtering

print("\n" + "=" * 80)
print("5. Модель 1: User-Based Collaborative Filtering")
print("=" * 80 + "\n")

# """
# User-Based CF находит похожих пользователей и рекомендует аниме,
# которые понравились им, но не были просмотрены целевым пользователем.
# 
# Алгоритм:
# 1. Создаем матрицу пользователь-аниме
# 2. Вычисляем схожесть между пользователями (cosine similarity)
# 3. Для предсказания оценки берем взвешенное среднее оценок похожих пользователей
# """

# Создаем матрицу пользователь-аниме для обучающей выборки
n_users = len(user_mapping)
n_anime = len(anime_mapping)

user_anime_matrix = np.zeros((n_users, n_anime))

for _, row in train_data.iterrows():
    user_anime_matrix[int(row['user_idx']), int(row['anime_idx'])] = row['rating']

print(f"Матрица пользователь-аниме: {user_anime_matrix.shape}")
print(f"Заполненность матрицы: {(user_anime_matrix > 0).sum() / user_anime_matrix.size * 100:.2f}%")

# Вычисляем схожесть между пользователями
print("\nВычисление схожести между пользователями...")
# Используем только подвыборку пользователей для ускорения
sample_size = min(1000, n_users)
sample_indices = np.random.choice(n_users, sample_size, replace=False)
user_similarity = cosine_similarity(user_anime_matrix[sample_indices])

print(f"Матрица схожести: {user_similarity.shape}")

# Функция для предсказания оценок
def predict_user_based(user_idx, anime_idx, k=10):
    """
    Предсказывает оценку пользователя для аниме на основе похожих пользователей
    
    k - количество ближайших соседей
    """
    if user_idx not in sample_indices:
        # Если пользователь не в выборке, возвращаем среднюю оценку
        return user_anime_matrix[:, anime_idx][user_anime_matrix[:, anime_idx] > 0].mean()
    
    user_sample_idx = np.where(sample_indices == user_idx)[0][0]
    
    # Находим k наиболее похожих пользователей
    similar_users_idx = np.argsort(user_similarity[user_sample_idx])[-k-1:-1][::-1]
    similar_users = sample_indices[similar_users_idx]
    
    # Берем оценки этих пользователей для данного аниме
    ratings = []
    similarities = []
    
    for i, sim_user in enumerate(similar_users):
        if user_anime_matrix[sim_user, anime_idx] > 0:
            ratings.append(user_anime_matrix[sim_user, anime_idx])
            similarities.append(user_similarity[user_sample_idx, similar_users_idx[i]])
    
    if len(ratings) == 0:
        # Если нет оценок от похожих пользователей, возвращаем среднюю
        return user_anime_matrix[:, anime_idx][user_anime_matrix[:, anime_idx] > 0].mean()
    
    # Взвешенное среднее
    return np.average(ratings, weights=similarities)

# Тестируем на небольшой выборке
print("\nТестирование User-Based CF на выборке...")
test_sample = test_data[test_data['user_idx'].isin(sample_indices)].sample(min(1000, len(test_data)))

predictions_ub = []
actuals_ub = []

for _, row in test_sample.iterrows():
    pred = predict_user_based(int(row['user_idx']), int(row['anime_idx']))
    if not np.isnan(pred):
        predictions_ub.append(pred)
        actuals_ub.append(row['rating'])

rmse_ub = np.sqrt(mean_squared_error(actuals_ub, predictions_ub))
mae_ub = mean_absolute_error(actuals_ub, predictions_ub)

print(f"\nUser-Based CF:")
print(f"  RMSE: {rmse_ub:.4f}")
print(f"  MAE: {mae_ub:.4f}")
print(f"  Протестировано: {len(predictions_ub)} оценок")

# ## Модель 2: Item-Based Collaborative Filtering

print("\n" + "=" * 80)
print("6. Модель 2: Item-Based Collaborative Filtering")
print("=" * 80 + "\n")

# """
# Item-Based CF находит похожие аниме и рекомендует их на основе того,
# что пользователь смотрел и оценил раньше.
# 
# Преимущества перед User-Based:
# - Схожесть между аниме более стабильна, чем между пользователями
# - Быстрее для больших датасетов с множеством пользователей
# """

# Вычисляем схожесть между аниме (транспонируем матрицу)
print("Вычисление схожести между аниме...")
# Используем подвыборку аниме
anime_sample_size = min(500, n_anime)
anime_sample_indices = np.random.choice(n_anime, anime_sample_size, replace=False)
anime_similarity = cosine_similarity(user_anime_matrix[:, anime_sample_indices].T)

print(f"Матрица схожести аниме: {anime_similarity.shape}")

# Функция для предсказания оценок
def predict_item_based(user_idx, anime_idx, k=10):
    """
    Предсказывает оценку на основе похожих аниме
    """
    if anime_idx not in anime_sample_indices:
        # Если аниме не в выборке, возвращаем среднюю оценку пользователя
        user_ratings = user_anime_matrix[user_idx][user_anime_matrix[user_idx] > 0]
        return user_ratings.mean() if len(user_ratings) > 0 else 7.0
    
    anime_sample_idx = np.where(anime_sample_indices == anime_idx)[0][0]
    
    # Находим k наиболее похожих аниме
    similar_anime_idx = np.argsort(anime_similarity[anime_sample_idx])[-k-1:-1][::-1]
    similar_anime = anime_sample_indices[similar_anime_idx]
    
    # Берем оценки пользователя для этих аниме
    ratings = []
    similarities = []
    
    for i, sim_anime in enumerate(similar_anime):
        if user_anime_matrix[user_idx, sim_anime] > 0:
            ratings.append(user_anime_matrix[user_idx, sim_anime])
            similarities.append(anime_similarity[anime_sample_idx, similar_anime_idx[i]])
    
    if len(ratings) == 0:
        # Если пользователь не оценивал похожие аниме
        user_ratings = user_anime_matrix[user_idx][user_anime_matrix[user_idx] > 0]
        return user_ratings.mean() if len(user_ratings) > 0 else 7.0
    
    # Взвешенное среднее
    return np.average(ratings, weights=similarities)

# Тестируем
print("\nТестирование Item-Based CF на выборке...")
test_sample_ib = test_data[test_data['anime_idx'].isin(anime_sample_indices)].sample(min(1000, len(test_data)))

predictions_ib = []
actuals_ib = []

for _, row in test_sample_ib.iterrows():
    pred = predict_item_based(int(row['user_idx']), int(row['anime_idx']))
    if not np.isnan(pred):
        predictions_ib.append(pred)
        actuals_ib.append(row['rating'])

rmse_ib = np.sqrt(mean_squared_error(actuals_ib, predictions_ib))
mae_ib = mean_absolute_error(actuals_ib, predictions_ib)

print(f"\nItem-Based CF:")
print(f"  RMSE: {rmse_ib:.4f}")
print(f"  MAE: {mae_ib:.4f}")
print(f"  Протестировано: {len(predictions_ib)} оценок")

# ## Модель 3: Baseline - средняя оценка

print("\n" + "=" * 80)
print("7. Baseline модель")
print("=" * 80 + "\n")

# """
# Простая baseline модель: предсказываем среднюю оценку для каждого аниме
# Это помогает оценить, насколько наши модели лучше простого подхода
# """

# Средние оценки для каждого аниме на обучающей выборке
anime_mean_ratings = train_data.groupby('anime_idx')['rating'].mean().to_dict()
global_mean = train_data['rating'].mean()

# Тестируем
test_sample_baseline = test_data.sample(min(1000, len(test_data)))
predictions_baseline = []
actuals_baseline = []

for _, row in test_sample_baseline.iterrows():
    anime_idx = int(row['anime_idx'])
    pred = anime_mean_ratings.get(anime_idx, global_mean)
    predictions_baseline.append(pred)
    actuals_baseline.append(row['rating'])

rmse_baseline = np.sqrt(mean_squared_error(actuals_baseline, predictions_baseline))
mae_baseline = mean_absolute_error(actuals_baseline, predictions_baseline)

print(f"Baseline (средняя оценка аниме):")
print(f"  RMSE: {rmse_baseline:.4f}")
print(f"  MAE: {mae_baseline:.4f}")

# ## Сравнение моделей

print("\n" + "=" * 80)
print("8. Сравнение моделей")
print("=" * 80 + "\n")

results = pd.DataFrame({
    'Model': ['Baseline', 'User-Based CF', 'Item-Based CF'],
    'RMSE': [rmse_baseline, rmse_ub, rmse_ib],
    'MAE': [mae_baseline, mae_ub, mae_ib]
})

print(results.to_string(index=False))
results.to_csv('../lab4/results_recommendations.csv', index=False)
print("\n✓ Результаты сохранены в 'results_recommendations.csv'")

# ## Визуализация

print("\n" + "=" * 80)
print("9. Создание визуализаций")
print("=" * 80 + "\n")

# График 1: Сравнение метрик
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = results['Model'].tolist()
x = np.arange(len(models))
width = 0.6

axes[0].bar(x, results['RMSE'], width, alpha=0.8, color=['gray', 'steelblue', 'coral'])
axes[0].set_xlabel('Модель')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Сравнение RMSE')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=15, ha='right')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x, results['MAE'], width, alpha=0.8, color=['gray', 'steelblue', 'coral'])
axes[1].set_xlabel('Модель')
axes[1].set_ylabel('MAE')
axes[1].set_title('Сравнение MAE')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=15, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../lab4/comparison_recommendations.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'comparison_recommendations.png'")
plt.close()

# График 2: Распределение оценок
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Распределение всех оценок
axes[0, 0].hist(ratings_df['rating'], bins=10, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Оценка')
axes[0, 0].set_ylabel('Частота')
axes[0, 0].set_title('Распределение оценок (все данные)')
axes[0, 0].grid(alpha=0.3)

# Активность пользователей
user_counts = ratings_filtered.groupby('user_id').size()
axes[0, 1].hist(user_counts, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Количество оценок')
axes[0, 1].set_ylabel('Количество пользователей')
axes[0, 1].set_title('Распределение активности пользователей')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# Популярность аниме
anime_counts = ratings_filtered.groupby('anime_id').size()
axes[1, 0].hist(anime_counts, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Количество оценок')
axes[1, 0].set_ylabel('Количество аниме')
axes[1, 0].set_title('Распределение популярности аниме')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3)

# Предсказания vs реальные (Item-Based)
axes[1, 1].scatter(actuals_ib, predictions_ib, alpha=0.3, s=10)
axes[1, 1].plot([0, 10], [0, 10], 'r--', lw=2)
axes[1, 1].set_xlabel('Реальная оценка')
axes[1, 1].set_ylabel('Предсказанная оценка')
axes[1, 1].set_title(f'Item-Based CF: Предсказания vs Реальные\nRMSE={rmse_ib:.4f}')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../lab4/recommendation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'recommendation_analysis.png'")
plt.close()

# ## Пример рекомендаций

print("\n" + "=" * 80)
print("10. Пример работы рекомендательной системы")
print("=" * 80 + "\n")

# Выберем случайного пользователя из обучающей выборки
sample_user_id = train_data['user_id'].sample(1).values[0]
sample_user_idx = user_mapping[sample_user_id]

print(f"Пользователь ID: {sample_user_id}")

# Аниме, которые пользователь уже оценил
user_rated = train_data[train_data['user_id'] == sample_user_id]
print(f"\nПользователь оценил {len(user_rated)} аниме")

# Топ-5 любимых аниме пользователя
top_rated = user_rated.nlargest(5, 'rating')[['anime_id', 'rating']]
top_rated_with_names = top_rated.merge(anime_df[['anime_id', 'Name']], on='anime_id', how='left')
print("\nТоп-5 любимых аниме пользователя:")
for _, row in top_rated_with_names.iterrows():
    print(f"  {row['Name'][:50]:50} - Оценка: {row['rating']}")

# Найдем аниме для рекомендации (которые пользователь еще не смотрел)
rated_anime = set(user_rated['anime_id'].values)
all_anime = set(anime_df['anime_id'].values)
unrated_anime = list(all_anime - rated_anime)

# Ограничимся аниме из нашей выборки
unrated_anime_in_sample = [a for a in unrated_anime if a in {aid: idx for aid, idx in anime_mapping.items()}][:100]

print(f"\nГенерация рекомендаций для {len(unrated_anime_in_sample)} непросмотренных аниме...")

# Предсказываем оценки для непросмотренных аниме
recommendations = []
for anime_id in unrated_anime_in_sample[:50]:  # Ограничимся 50 для скорости
    if anime_id in anime_mapping:
        anime_idx = anime_mapping[anime_id]
        if anime_idx in anime_sample_indices:
            pred_rating = predict_item_based(sample_user_idx, anime_idx)
            if not np.isnan(pred_rating):
                recommendations.append({
                    'anime_id': anime_id,
                    'predicted_rating': pred_rating
                })

# Сортируем по предсказанному рейтингу
recommendations_df = pd.DataFrame(recommendations).sort_values('predicted_rating', ascending=False)
recommendations_df = recommendations_df.merge(anime_df[['anime_id', 'Name', 'Score', 'Genres']], 
                                               on='anime_id', how='left')

print("\nТоп-10 рекомендаций:")
for i, row in recommendations_df.head(10).iterrows():
    print(f"  {row['Name'][:45]:45} | Предсказ: {row['predicted_rating']:.2f} | Реальный: {row['Score']:.2f} | {row['Genres'][:30]}")

# ## Итоговый вывод

print("\n" + "=" * 80)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 80 + "\n")

best_model = results.loc[results['RMSE'].idxmin()]

print(f"""
ЭКСПЕРИМЕНТ 3: Рекомендательная система

1. СРАВНЕНИЕ ПОДХОДОВ:
   - Лучшая модель: {best_model['Model']}
   - RMSE: {best_model['RMSE']:.4f}
   - MAE: {best_model['MAE']:.4f}

2. РЕЗУЛЬТАТЫ МОДЕЛЕЙ:
   - Baseline (средняя оценка): RMSE={rmse_baseline:.4f}, MAE={mae_baseline:.4f}
   - User-Based CF: RMSE={rmse_ub:.4f}, MAE={mae_ub:.4f}
   - Item-Based CF: RMSE={rmse_ib:.4f}, MAE={mae_ib:.4f}

3. ВЫВОДЫ:
   - Collaborative Filtering показывает результаты {"лучше" if min(rmse_ub, rmse_ib) < rmse_baseline else "сопоставимые с"} baseline
   - Item-Based подход {"более эффективен" if rmse_ib < rmse_ub else "показывает результаты сопоставимые с User-Based"}
   - Ошибка предсказания составляет ±{best_model['MAE']:.2f} балла от реальной оценки

4. ОГРАНИЧЕНИЯ:
   - Проблема холодного старта: сложно давать рекомендации новым пользователям
   - Разреженность данных: большинство пользователей оценили мало аниме
   - Вычислительная сложность: для полного датасета требуется больше ресурсов

5. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:
   - Система может использоваться для персонализированных рекомендаций на сайтах
   - Item-Based CF более стабилен и быстрее для продакшена
   - Гибридный подход (CF + Content-Based) может улучшить качество

6. ВОЗМОЖНЫЕ УЛУЧШЕНИЯ:
   - Использование матричной факторизации (SVD, ALS)
   - Глубокое обучение (нейронные сети для рекомендаций)
   - Учет временной динамики (более свежие оценки важнее)
   - Гибридная система: CF + признаки аниме (жанры, студия, и т.д.)
""")

print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ 3 ЗАВЕРШЕН")
print("=" * 80)
