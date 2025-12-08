# ========================================
# ЛАБОРАТОРНАЯ РАБОТА 4: ПРОГНОЗИРОВАНИЕ РЕЙТИНГА АНИМЕ
# Эксперимент 2: Предсказание рейтинга на основе данных, известных до релиза
# ========================================

# ## Импорт необходимых библиотек

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ## Загрузка данных

print("=" * 80)
print("ЭКСПЕРИМЕНТ 2: ПРЕДСКАЗАНИЕ РЕЙТИНГА БЕЗ POST-RELEASE ПАРАМЕТРОВ")
print("=" * 80)
print("\n1. Загрузка данных...\n")

# """
# Цель: предсказать рейтинг аниме ДО его релиза, используя только
# информацию, которая известна заранее (жанры, студия, тип, источник и т.д.)
# 
# НЕ используем:
# - Favorites (добавляются после просмотра)
# - Scored By (количество оценок - известно после релиза)
# - Members (количество участников - растет со временем)
# - Rank (зависит от Score)
# - Popularity (зависит от Members)
# """

df = pd.read_csv('../data/processed/anime_processed.csv')

print(f"Размер датасета: {df.shape}")

# ## Выбор признаков

print("\n" + "=" * 80)
print("2. Подготовка признаков (только pre-release данные)")
print("=" * 80 + "\n")

target = 'Score'
df_model = df[df[target].notna()].copy()

print(f"Количество записей с известным рейтингом: {len(df_model)}")

# Числовые признаки (доступные до релиза)
numeric_features = ['Episodes']  # Количество эпизодов обычно известно заранее

# Категориальные признаки
categorical_features = ['Type', 'Status', 'Source', 'Rating']

# Жанры - отдельная обработка (разделены запятыми)
# Извлекаем наиболее популярные жанры
print("\nИзвлечение информации о жанрах...")

# Разделяем жанры и считаем частоту
all_genres = []
for genres_str in df_model['Genres'].dropna():
    genres_list = [g.strip() for g in str(genres_str).split(',')]
    all_genres.extend(genres_list)

from collections import Counter
genre_counts = Counter(all_genres)
top_genres = [genre for genre, count in genre_counts.most_common(20)]

print(f"Топ-20 жанров: {top_genres[:10]}...")

# Создаем бинарные признаки для топ-жанров
for genre in top_genres:
    df_model[f'Genre_{genre}'] = df_model['Genres'].apply(
        lambda x: 1 if genre in str(x) else 0
    )

genre_features = [f'Genre_{genre}' for genre in top_genres]

print(f"\nЧисловые признаки ({len(numeric_features)}): {numeric_features}")
print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")
print(f"Признаки жанров ({len(genre_features)}): создано {len(genre_features)} бинарных признаков")

# ## Обработка пропущенных значений

print("\n" + "=" * 80)
print("3. Обработка пропущенных значений")
print("=" * 80 + "\n")

# Заполняем пропуски в Episodes медианой
if df_model['Episodes'].isna().sum() > 0:
    print(f"Пропуски в Episodes: {df_model['Episodes'].isna().sum()}")
    df_model['Episodes'].fillna(df_model['Episodes'].median(), inplace=True)

for col in categorical_features:
    missing = df_model[col].isna().sum()
    if missing > 0:
        print(f"Пропуски в {col}: {missing}")
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)

# ## Кодирование категориальных признаков

print("\n" + "=" * 80)
print("4. Кодирование категориальных признаков")
print("=" * 80 + "\n")

# LabelEncoder для Rating (порядковый признак)
rating_order = ['G - All Ages', 'PG - Children', 'PG-13 - Teens 13 or older', 
                'R - 17+ (violence & profanity)', 'R+ - Mild Nudity', 'Rx - Hentai']
le_rating = LabelEncoder()
le_rating.fit(rating_order)
df_model['Rating_encoded'] = df_model['Rating'].apply(
    lambda x: le_rating.transform([x])[0] if x in rating_order else -1
)

print("Кодирование Rating (LabelEncoder)")

# One-Hot Encoding для остальных категориальных признаков
categorical_to_encode = ['Type', 'Status', 'Source']
df_encoded = pd.get_dummies(df_model, columns=categorical_to_encode, prefix=categorical_to_encode)

print(f"One-Hot Encoding применен к: {categorical_to_encode}")

# ## Формирование итогового набора признаков

feature_columns = numeric_features + ['Rating_encoded'] + genre_features
one_hot_columns = [col for col in df_encoded.columns if any(col.startswith(prefix + '_') for prefix in categorical_to_encode)]
feature_columns.extend(one_hot_columns)

X = df_encoded[feature_columns]
y = df_encoded[target]

print(f"\nИтоговое количество признаков: {len(feature_columns)}")
print(f"  - Числовые: {len(numeric_features)}")
print(f"  - Rating (encoded): 1")
print(f"  - Жанры: {len(genre_features)}")
print(f"  - One-Hot encoded: {len(one_hot_columns)}")
print(f"\nРазмер X: {X.shape}, размер y: {y.shape}")

# ## Стандартизация числовых признаков

print("\n" + "=" * 80)
print("5. Стандартизация числовых признаков")
print("=" * 80 + "\n")

scaler = StandardScaler()
X_scaled = X.copy()

# Масштабируем только Episodes
numeric_indices = [feature_columns.index(col) for col in numeric_features]
X_scaled.iloc[:, numeric_indices] = scaler.fit_transform(X.iloc[:, numeric_indices])

print("Стандартизация применена к: Episodes")

# ## Разделение данных

print("\n" + "=" * 80)
print("6. Разделение данных на train/test")
print("=" * 80 + "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"\nРаспределение Score:")
print(f"  Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
print(f"  Test:  mean={y_test.mean():.2f}, std={y_test.std():.2f}")

# ## Обучение моделей

print("\n" + "=" * 80)
print("7. Обучение и сравнение моделей")
print("=" * 80 + "\n")

results = {}

# ### Модель 1: Линейная регрессия

print("\n--- Модель 1: Линейная регрессия ---")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_train_lr = lr_model.predict(X_train)
y_pred_test_lr = lr_model.predict(X_test)

lr_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
lr_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))
lr_mae_train = mean_absolute_error(y_train, y_pred_train_lr)
lr_mae_test = mean_absolute_error(y_test, y_pred_test_lr)
lr_r2_train = r2_score(y_train, y_pred_train_lr)
lr_r2_test = r2_score(y_test, y_pred_test_lr)

print(f"RMSE: Train={lr_rmse_train:.4f}, Test={lr_rmse_test:.4f}")
print(f"MAE:  Train={lr_mae_train:.4f}, Test={lr_mae_test:.4f}")
print(f"R²:   Train={lr_r2_train:.4f}, Test={lr_r2_test:.4f}")

results['Linear Regression'] = {
    'model': lr_model,
    'rmse_train': lr_rmse_train,
    'rmse_test': lr_rmse_test,
    'mae_train': lr_mae_train,
    'mae_test': lr_mae_test,
    'r2_train': lr_r2_train,
    'r2_test': lr_r2_test,
    'predictions': y_pred_test_lr
}

# ### Модель 2: Дерево решений

print("\n--- Модель 2: Дерево решений ---")

dt_model = DecisionTreeRegressor(max_depth=8, min_samples_split=30, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_train_dt = dt_model.predict(X_train)
y_pred_test_dt = dt_model.predict(X_test)

dt_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_dt))
dt_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_dt))
dt_mae_train = mean_absolute_error(y_train, y_pred_train_dt)
dt_mae_test = mean_absolute_error(y_test, y_pred_test_dt)
dt_r2_train = r2_score(y_train, y_pred_train_dt)
dt_r2_test = r2_score(y_test, y_pred_test_dt)

print(f"RMSE: Train={dt_rmse_train:.4f}, Test={dt_rmse_test:.4f}")
print(f"MAE:  Train={dt_mae_train:.4f}, Test={dt_mae_test:.4f}")
print(f"R²:   Train={dt_r2_train:.4f}, Test={dt_r2_test:.4f}")

results['Decision Tree'] = {
    'model': dt_model,
    'rmse_train': dt_rmse_train,
    'rmse_test': dt_rmse_test,
    'mae_train': dt_mae_train,
    'mae_test': dt_mae_test,
    'r2_train': dt_r2_train,
    'r2_test': dt_r2_test,
    'predictions': y_pred_test_dt
}

# ### Модель 3: Random Forest

print("\n--- Модель 3: Random Forest ---")

rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=15,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

rf_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
rf_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))
rf_mae_train = mean_absolute_error(y_train, y_pred_train_rf)
rf_mae_test = mean_absolute_error(y_test, y_pred_test_rf)
rf_r2_train = r2_score(y_train, y_pred_train_rf)
rf_r2_test = r2_score(y_test, y_pred_test_rf)

print(f"RMSE: Train={rf_rmse_train:.4f}, Test={rf_rmse_test:.4f}")
print(f"MAE:  Train={rf_mae_train:.4f}, Test={rf_mae_test:.4f}")
print(f"R²:   Train={rf_r2_train:.4f}, Test={rf_r2_test:.4f}")

results['Random Forest'] = {
    'model': rf_model,
    'rmse_train': rf_rmse_train,
    'rmse_test': rf_rmse_test,
    'mae_train': rf_mae_train,
    'mae_test': rf_mae_test,
    'r2_train': rf_r2_train,
    'r2_test': rf_r2_test,
    'predictions': y_pred_test_rf
}

# ### Модель 4: Gradient Boosting

print("\n--- Модель 4: Gradient Boosting ---")

gb_model = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.05,
                                      random_state=42)
gb_model.fit(X_train, y_train)

y_pred_train_gb = gb_model.predict(X_train)
y_pred_test_gb = gb_model.predict(X_test)

gb_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_gb))
gb_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_gb))
gb_mae_train = mean_absolute_error(y_train, y_pred_train_gb)
gb_mae_test = mean_absolute_error(y_test, y_pred_test_gb)
gb_r2_train = r2_score(y_train, y_pred_train_gb)
gb_r2_test = r2_score(y_test, y_pred_test_gb)

print(f"RMSE: Train={gb_rmse_train:.4f}, Test={gb_rmse_test:.4f}")
print(f"MAE:  Train={gb_mae_train:.4f}, Test={gb_mae_test:.4f}")
print(f"R²:   Train={gb_r2_train:.4f}, Test={gb_r2_test:.4f}")

results['Gradient Boosting'] = {
    'model': gb_model,
    'rmse_train': gb_rmse_train,
    'rmse_test': gb_rmse_test,
    'mae_train': gb_mae_train,
    'mae_test': gb_mae_test,
    'r2_train': gb_r2_train,
    'r2_test': gb_r2_test,
    'predictions': y_pred_test_gb
}

# ## Сравнение моделей

print("\n" + "=" * 80)
print("8. Сводная таблица результатов")
print("=" * 80 + "\n")

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE_Train': [results[m]['rmse_train'] for m in results.keys()],
    'RMSE_Test': [results[m]['rmse_test'] for m in results.keys()],
    'MAE_Train': [results[m]['mae_train'] for m in results.keys()],
    'MAE_Test': [results[m]['mae_test'] for m in results.keys()],
    'R2_Train': [results[m]['r2_train'] for m in results.keys()],
    'R2_Test': [results[m]['r2_test'] for m in results.keys()],
})

print(comparison_df.to_string(index=False))

comparison_df.to_csv('../lab4/results_pre_release.csv', index=False)
print("\n✓ Результаты сохранены в 'results_pre_release.csv'")

# ## Кросс-валидация

print("\n" + "=" * 80)
print("9. Кросс-валидация")
print("=" * 80 + "\n")

cv_results = {}

for name, result in results.items():
    model = result['model']
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='neg_mean_squared_error', n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores)
    cv_results[name] = {
        'mean_cv_rmse': cv_rmse.mean(),
        'std_cv_rmse': cv_rmse.std()
    }
    print(f"{name}:")
    print(f"  CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")

# ## Анализ важности признаков

print("\n" + "=" * 80)
print("10. Анализ важности признаков (Random Forest)")
print("=" * 80 + "\n")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Топ-20 наиболее важных признаков:")
print(feature_importance.head(20).to_string(index=False))

feature_importance.to_csv('../lab4/feature_importance_pre_release.csv', index=False)

# Анализ важности по категориям
print("\nВажность по категориям:")
genre_importance = feature_importance[feature_importance['feature'].str.startswith('Genre_')]['importance'].sum()
type_importance = feature_importance[feature_importance['feature'].str.startswith('Type_')]['importance'].sum()
source_importance = feature_importance[feature_importance['feature'].str.startswith('Source_')]['importance'].sum()
status_importance = feature_importance[feature_importance['feature'].str.startswith('Status_')]['importance'].sum()
episodes_importance = feature_importance[feature_importance['feature'] == 'Episodes']['importance'].sum()
rating_importance = feature_importance[feature_importance['feature'] == 'Rating_encoded']['importance'].sum()

print(f"  Жанры: {genre_importance:.4f}")
print(f"  Тип (Type): {type_importance:.4f}")
print(f"  Источник (Source): {source_importance:.4f}")
print(f"  Статус: {status_importance:.4f}")
print(f"  Количество эпизодов: {episodes_importance:.4f}")
print(f"  Возрастной рейтинг: {rating_importance:.4f}")

# ## Визуализация

print("\n" + "=" * 80)
print("11. Создание визуализаций")
print("=" * 80 + "\n")

# График 1: Сравнение метрик
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = list(results.keys())
rmse_train = [results[m]['rmse_train'] for m in models]
rmse_test = [results[m]['rmse_test'] for m in models]
mae_train = [results[m]['mae_train'] for m in models]
mae_test = [results[m]['mae_test'] for m in models]
r2_train = [results[m]['r2_train'] for m in models]
r2_test = [results[m]['r2_test'] for m in models]

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, rmse_train, width, label='Train', alpha=0.8)
axes[0].bar(x + width/2, rmse_test, width, label='Test', alpha=0.8)
axes[0].set_xlabel('Модель')
axes[0].set_ylabel('RMSE')
axes[0].set_title('Сравнение RMSE (Pre-Release признаки)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x - width/2, mae_train, width, label='Train', alpha=0.8)
axes[1].bar(x + width/2, mae_test, width, label='Test', alpha=0.8)
axes[1].set_xlabel('Модель')
axes[1].set_ylabel('MAE')
axes[1].set_title('Сравнение MAE (Pre-Release признаки)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(x - width/2, r2_train, width, label='Train', alpha=0.8)
axes[2].bar(x + width/2, r2_test, width, label='Test', alpha=0.8)
axes[2].set_xlabel('Модель')
axes[2].set_ylabel('R²')
axes[2].set_title('Сравнение R² (Pre-Release признаки)')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=45, ha='right')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../lab4/comparison_metrics_pre_release.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'comparison_metrics_pre_release.png'")
plt.close()

# График 2: Важность признаков по категориям
categories = ['Жанры', 'Тип', 'Источник', 'Статус', 'Эпизоды', 'Рейтинг']
importances = [genre_importance, type_importance, source_importance, 
               status_importance, episodes_importance, rating_importance]

plt.figure(figsize=(10, 6))
plt.bar(categories, importances, alpha=0.8, color='steelblue')
plt.xlabel('Категория признаков')
plt.ylabel('Суммарная важность')
plt.title('Важность категорий признаков (Random Forest)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../lab4/category_importance_pre_release.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'category_importance_pre_release.png'")
plt.close()

# График 3: Топ жанров по важности
top_genre_features = feature_importance[feature_importance['feature'].str.startswith('Genre_')].head(15)
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_genre_features)), top_genre_features['importance'])
plt.yticks(range(len(top_genre_features)), 
           [f.replace('Genre_', '') for f in top_genre_features['feature']])
plt.xlabel('Важность')
plt.ylabel('Жанр')
plt.title('Топ-15 жанров по важности')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../lab4/top_genres_importance.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'top_genres_importance.png'")
plt.close()

# График 4: Предсказания vs реальные значения
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (name, result) in enumerate(results.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    y_pred = result['predictions']
    ax.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Реальный рейтинг')
    ax.set_ylabel('Предсказанный рейтинг')
    ax.set_title(f'{name}\nRMSE={result["rmse_test"]:.4f}, R²={result["r2_test"]:.4f}')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../lab4/predictions_vs_actual_pre_release.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'predictions_vs_actual_pre_release.png'")
plt.close()

# ## Анализ ошибок

print("\n" + "=" * 80)
print("12. Анализ ошибок")
print("=" * 80 + "\n")

best_model_name = min(results.keys(), key=lambda x: results[x]['rmse_test'])
best_predictions = results[best_model_name]['predictions']
best_errors = y_test - best_predictions

print(f"Лучшая модель: {best_model_name}")
print(f"\nСтатистика ошибок:")
print(f"  Средняя ошибка: {best_errors.mean():.4f}")
print(f"  Медианная ошибка: {best_errors.median():.4f}")
print(f"  Стандартное отклонение: {best_errors.std():.4f}")
print(f"  Минимальная ошибка: {best_errors.min():.4f}")
print(f"  Максимальная ошибка: {best_errors.max():.4f}")

# ## Сравнение с Экспериментом 1

print("\n" + "=" * 80)
print("13. Сравнение с Экспериментом 1 (Full Features)")
print("=" * 80 + "\n")

# Загружаем результаты первого эксперимента
exp1_results = pd.read_csv('../lab4/results_full_features.csv')

comparison = pd.DataFrame({
    'Model': results.keys(),
    'Exp1_RMSE': exp1_results['RMSE_Test'].values,
    'Exp2_RMSE': [results[m]['rmse_test'] for m in results.keys()],
    'Exp1_R2': exp1_results['R2_Test'].values,
    'Exp2_R2': [results[m]['r2_test'] for m in results.keys()],
    'RMSE_Diff': [results[m]['rmse_test'] - exp1_results.loc[i, 'RMSE_Test'] 
                  for i, m in enumerate(results.keys())],
    'R2_Diff': [results[m]['r2_test'] - exp1_results.loc[i, 'R2_Test'] 
                for i, m in enumerate(results.keys())]
})

print(comparison.to_string(index=False))

# ## Итоговый вывод

print("\n" + "=" * 80)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 80 + "\n")

print(f"""
ЭКСПЕРИМЕНТ 2: Предсказание рейтинга на основе pre-release данных

1. КАЧЕСТВО МОДЕЛЕЙ:
   - Лучшая модель: {best_model_name}
   - RMSE на тестовой выборке: {results[best_model_name]['rmse_test']:.4f}
   - MAE на тестовой выборке: {results[best_model_name]['mae_test']:.4f}
   - R² на тестовой выборке: {results[best_model_name]['r2_test']:.4f}

2. СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ 1:
   - Ухудшение RMSE: +{comparison.loc[comparison['Model'] == best_model_name, 'RMSE_Diff'].values[0]:.4f}
   - Ухудшение R²: {comparison.loc[comparison['Model'] == best_model_name, 'R2_Diff'].values[0]:.4f}
   - Это ожидаемо, так как мы не используем важные пост-релизные метрики

3. ВАЖНЫЕ ПРИЗНАКИ ДЛЯ PRE-RELEASE ПРЕДСКАЗАНИЯ:
   Топ-5 признаков: {', '.join(feature_importance.head(5)['feature'].tolist())}
   
   Наиболее важные категории:
   - Жанры: {genre_importance:.2%}
   - Тип аниме: {type_importance:.2%}
   - Источник: {source_importance:.2%}

4. ВЫВОДЫ:
   - Жанры, тип и источник - главные факторы для предсказания рейтинга до релиза
   - Модели на основе ансамблей (RF, GB) показывают лучшие результаты
   - Качество предсказания умеренное (R² ~ {results[best_model_name]['r2_test']:.2f}), что логично
     без информации о популярности

5. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:
   - Модель может использоваться для предварительной оценки потенциала аниме
   - Полезна для студий и продюсеров на этапе планирования
   - Помогает понять, какие факторы (жанр, тип) влияют на успешность
""")

print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ 2 ЗАВЕРШЕН")
print("=" * 80)
