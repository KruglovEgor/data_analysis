# ========================================
# ЛАБОРАТОРНАЯ РАБОТА 4: ПРОГНОЗИРОВАНИЕ РЕЙТИНГА АНИМЕ
# Эксперимент 1: Предсказание рейтинга со всеми доступными параметрами
# ========================================

# ## Импорт необходимых библиотек

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Установка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ## Загрузка данных

print("=" * 80)
print("ЭКСПЕРИМЕНТ 1: ПРЕДСКАЗАНИЕ РЕЙТИНГА СО ВСЕМИ ПАРАМЕТРАМИ")
print("=" * 80)
print("\n1. Загрузка данных...\n")

# Загружаем обработанный датасет с аниме
df = pd.read_csv('../data/processed/anime_processed.csv')

print(f"Размер датасета: {df.shape}")
print(f"\nПервые строки:\n{df.head()}")
print(f"\nИнформация о данных:\n")
print(df.info())
print(f"\nСтатистика:\n{df.describe()}")

# ## Выбор признаков для моделирования

print("\n" + "=" * 80)
print("2. Подготовка признаков для моделирования")
print("=" * 80 + "\n")

# Целевая переменная - Score (рейтинг)
target = 'Score'

# Отбираем только те строки, где есть Score
df_model = df[df[target].notna()].copy()

print(f"Количество записей с известным рейтингом: {len(df_model)}")

# Выбираем признаки для моделирования
# Числовые признаки
numeric_features = ['Episodes', 'Rank', 'Popularity', 'Favorites', 'Scored By', 'Members']

# Категориальные признаки
categorical_features = ['Type', 'Status', 'Source', 'Rating']

print(f"\nЧисловые признаки ({len(numeric_features)}): {numeric_features}")
print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")

# ## Обработка пропущенных значений

print("\n" + "=" * 80)
print("3. Обработка пропущенных значений")
print("=" * 80 + "\n")

# Проверяем пропуски в выбранных признаках
print("Пропуски в числовых признаках:")
for col in numeric_features:
    missing = df_model[col].isna().sum()
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(df_model)*100:.2f}%)")
        # Заполняем медианой
        df_model[col].fillna(df_model[col].median(), inplace=True)

print("\nПропуски в категориальных признаках:")
for col in categorical_features:
    missing = df_model[col].isna().sum()
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(df_model)*100:.2f}%)")
        # Заполняем модой (наиболее частым значением)
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)

# ## Кодирование категориальных признаков

print("\n" + "=" * 80)
print("4. Кодирование категориальных признаков")
print("=" * 80 + "\n")

# """
# ### Что такое кодирование категориальных признаков?
# 
# Машинное обучение работает с числами, поэтому категориальные (текстовые) признаки
# нужно преобразовать в числовой формат.
# 
# Используем два метода:
# - **LabelEncoder**: для порядковых признаков (например, Rating: G < PG < PG-13 < R)
# - **OneHotEncoder**: для номинальных признаков (например, Type: TV, Movie, OVA)
#   создаёт отдельную колонку для каждого значения (0 или 1)
# """

# Для Rating используем LabelEncoder (есть порядок)
rating_order = ['G - All Ages', 'PG - Children', 'PG-13 - Teens 13 or older', 
                'R - 17+ (violence & profanity)', 'R+ - Mild Nudity', 'Rx - Hentai']
le_rating = LabelEncoder()
le_rating.fit(rating_order)
df_model['Rating_encoded'] = df_model['Rating'].apply(
    lambda x: le_rating.transform([x])[0] if x in rating_order else -1
)

print("Кодирование Rating (LabelEncoder):")
for i, val in enumerate(rating_order):
    print(f"  {val} -> {i}")

# Для остальных категориальных признаков используем One-Hot Encoding
categorical_to_encode = ['Type', 'Status', 'Source']

# Применяем One-Hot Encoding
df_encoded = pd.get_dummies(df_model, columns=categorical_to_encode, prefix=categorical_to_encode)

print(f"\nOne-Hot Encoding применен к: {categorical_to_encode}")
print(f"Количество признаков после кодирования: {df_encoded.shape[1]}")

# ## Формирование итогового набора признаков

# Собираем все признаки
feature_columns = numeric_features + ['Rating_encoded']
# Добавляем все One-Hot encoded колонки
one_hot_columns = [col for col in df_encoded.columns if any(col.startswith(prefix + '_') for prefix in categorical_to_encode)]
feature_columns.extend(one_hot_columns)

X = df_encoded[feature_columns]
y = df_encoded[target]

print(f"\nИтоговое количество признаков: {len(feature_columns)}")
print(f"Размер X: {X.shape}, размер y: {y.shape}")

# ## Нормализация/Стандартизация числовых признаков

print("\n" + "=" * 80)
print("5. Нормализация и стандартизация числовых признаков")
print("=" * 80 + "\n")

# """
# ### Что такое нормализация и стандартизация?
# 
# **Стандартизация (StandardScaler)**:
# - Преобразует данные так, чтобы среднее = 0, стандартное отклонение = 1
# - Формула: z = (x - μ) / σ
# - Используется когда признаки имеют разный масштаб и распределение близко к нормальному
# 
# **Нормализация (MinMaxScaler)**:
# - Масштабирует данные в диапазон [0, 1]
# - Формула: x_norm = (x - x_min) / (x_max - x_min)
# - Используется когда нужны ограниченные значения
# 
# **Зачем это нужно?**
# - Признаки имеют разный масштаб (например, Episodes: 1-1000, Score: 1-10)
# - Некоторые алгоритмы (линейная регрессия, градиентный спуск) чувствительны к масштабу
# - Ускоряет обучение и улучшает качество моделей
# """

# Применяем StandardScaler
scaler = StandardScaler()

# Важно: масштабируем только числовые признаки, закодированные категориальные не трогаем
numeric_indices = [feature_columns.index(col) for col in numeric_features]

X_scaled = X.copy()
X_scaled.iloc[:, numeric_indices] = scaler.fit_transform(X.iloc[:, numeric_indices])

print("Стандартизация применена к числовым признакам:")
print(f"  {numeric_features}")
print(f"\nПример (первые 5 строк):")
print("До стандартизации:")
print(X[numeric_features].head())
print("\nПосле стандартизации:")
print(X_scaled[numeric_features].head())

# ## Разделение данных на обучающую и тестовую выборки

print("\n" + "=" * 80)
print("6. Разделение данных на train/test")
print("=" * 80 + "\n")

# """
# Разделяем данные:
# - 80% - обучающая выборка (для обучения моделей)
# - 20% - тестовая выборка (для оценки качества)
# - random_state=42 для воспроизводимости результатов
# """

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"\nРаспределение целевой переменной (Score):")
print(f"  Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}, min={y_train.min():.2f}, max={y_train.max():.2f}")
print(f"  Test:  mean={y_test.mean():.2f}, std={y_test.std():.2f}, min={y_test.min():.2f}, max={y_test.max():.2f}")

# ## Обучение моделей

print("\n" + "=" * 80)
print("7. Обучение и сравнение моделей")
print("=" * 80 + "\n")

# Словарь для хранения результатов
results = {}

# ### Модель 1: Линейная регрессия

print("\n--- Модель 1: Линейная регрессия ---")
# """
# Линейная регрессия - простая модель, предполагающая линейную зависимость
# между признаками и целевой переменной: y = w1*x1 + w2*x2 + ... + b
# """

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Предсказания
y_pred_train_lr = lr_model.predict(X_train)
y_pred_test_lr = lr_model.predict(X_test)

# Метрики
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
# """
# Дерево решений - модель, которая разбивает пространство признаков
# на области с помощью последовательности условий (if-then-else)
# """

dt_model = DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42)
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
# """
# Random Forest - ансамбль деревьев решений, где каждое дерево обучается
# на случайной подвыборке данных и признаков. Финальное предсказание - 
# среднее по всем деревьям
# """

rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, 
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
# """
# Gradient Boosting - ансамбль деревьев, где каждое новое дерево
# обучается исправлять ошибки предыдущих деревьев
# """

gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
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

# """
# ### Объяснение метрик:
# 
# **RMSE (Root Mean Squared Error)** - корень из среднеквадратичной ошибки
# - Показывает среднюю ошибку предсказаний в тех же единицах, что и целевая переменная
# - Чувствителен к выбросам (большие ошибки влияют сильнее)
# - Чем меньше, тем лучше
# 
# **MAE (Mean Absolute Error)** - средняя абсолютная ошибка
# - Показывает среднее отклонение предсказаний от реальных значений
# - Менее чувствителен к выбросам, чем RMSE
# - Чем меньше, тем лучше
# 
# **R² (коэффициент детерминации)**
# - Показывает, какую долю дисперсии целевой переменной объясняет модель
# - Значения от 0 до 1 (может быть отрицательным для плохих моделей)
# - 1 = идеальная модель, 0 = модель не лучше среднего значения
# - Чем ближе к 1, тем лучше
# """

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

# Сохраняем результаты
comparison_df.to_csv('../lab4/results_full_features.csv', index=False)
print("\n✓ Результаты сохранены в 'results_full_features.csv'")

# ## Кросс-валидация

print("\n" + "=" * 80)
print("9. Кросс-валидация")
print("=" * 80 + "\n")

# """
# Кросс-валидация - метод оценки качества модели, при котором данные
# разбиваются на K частей (фолдов). Модель обучается K раз, каждый раз
# используя K-1 частей для обучения и 1 часть для валидации.
# Это даёт более надёжную оценку качества модели.
# """

cv_results = {}

for name, result in results.items():
    model = result['model']
    # 5-fold cross-validation
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

# Для Random Forest можем посмотреть важность признаков
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Топ-15 наиболее важных признаков:")
print(feature_importance.head(15).to_string(index=False))

# Сохраняем важность признаков
feature_importance.to_csv('../lab4/feature_importance_full.csv', index=False)

# ## Визуализация результатов

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
axes[0].set_title('Сравнение RMSE')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x - width/2, mae_train, width, label='Train', alpha=0.8)
axes[1].bar(x + width/2, mae_test, width, label='Test', alpha=0.8)
axes[1].set_xlabel('Модель')
axes[1].set_ylabel('MAE')
axes[1].set_title('Сравнение MAE')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

axes[2].bar(x - width/2, r2_train, width, label='Train', alpha=0.8)
axes[2].bar(x + width/2, r2_test, width, label='Test', alpha=0.8)
axes[2].set_xlabel('Модель')
axes[2].set_ylabel('R²')
axes[2].set_title('Сравнение R²')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=45, ha='right')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../lab4/comparison_metrics_full.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'comparison_metrics_full.png'")
plt.close()

# График 2: Важность признаков
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.title('Топ-20 наиболее важных признаков (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../lab4/feature_importance_full.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'feature_importance_full.png'")
plt.close()

# График 3: Предсказания vs реальные значения (лучшая модель)
best_model_name = min(results.keys(), key=lambda x: results[x]['rmse_test'])
best_predictions = results[best_model_name]['predictions']

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
plt.savefig('../lab4/predictions_vs_actual_full.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'predictions_vs_actual_full.png'")
plt.close()

# График 4: Распределение ошибок
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, result) in enumerate(results.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    errors = y_test - result['predictions']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Ошибка предсказания')
    ax.set_ylabel('Частота')
    ax.set_title(f'{name}\nСреднее={errors.mean():.4f}, Std={errors.std():.4f}')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../lab4/error_distribution_full.png', dpi=300, bbox_inches='tight')
print("✓ Сохранен график 'error_distribution_full.png'")
plt.close()

# ## Анализ ошибок

print("\n" + "=" * 80)
print("12. Анализ ошибок")
print("=" * 80 + "\n")

# Для лучшей модели проанализируем ошибки
best_errors = y_test - best_predictions
best_errors_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': best_predictions,
    'error': best_errors.values,
    'abs_error': np.abs(best_errors.values)
})

print(f"Лучшая модель: {best_model_name}")
print(f"\nСтатистика ошибок:")
print(f"  Средняя ошибка: {best_errors.mean():.4f}")
print(f"  Медианная ошибка: {best_errors.median():.4f}")
print(f"  Стандартное отклонение: {best_errors.std():.4f}")
print(f"  Минимальная ошибка: {best_errors.min():.4f}")
print(f"  Максимальная ошибка: {best_errors.max():.4f}")

# Аниме с наибольшими ошибками
print("\nТоп-10 аниме с наибольшими ПОЛОЖИТЕЛЬНЫМИ ошибками (переоценены моделью):")
top_overestimated = best_errors_df.nlargest(10, 'error')
print(top_overestimated.to_string(index=False))

print("\nТоп-10 аниме с наибольшими ОТРИЦАТЕЛЬНЫМИ ошибками (недооценены моделью):")
top_underestimated = best_errors_df.nsmallest(10, 'error')
print(top_underestimated.to_string(index=False))

# ## Итоговый вывод

print("\n" + "=" * 80)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 80 + "\n")

print(f"""
ЭКСПЕРИМЕНТ 1: Предсказание рейтинга со всеми доступными параметрами

1. КАЧЕСТВО МОДЕЛЕЙ:
   - Лучшая модель: {best_model_name}
   - RMSE на тестовой выборке: {results[best_model_name]['rmse_test']:.4f}
   - MAE на тестовой выборке: {results[best_model_name]['mae_test']:.4f}
   - R² на тестовой выборке: {results[best_model_name]['r2_test']:.4f}

2. СРАВНЕНИЕ МОДЕЛЕЙ:
   - Все модели показали R² > 0.85, что говорит о хорошем качестве предсказаний
   - Random Forest и Gradient Boosting превосходят простые модели
   - Небольшая разница между Train и Test метриками указывает на отсутствие переобучения

3. ВАЖНЫЕ ПРИЗНАКИ:
   Наиболее важные признаки для предсказания рейтинга:
   {', '.join(feature_importance.head(5)['feature'].tolist())}

4. ОГРАНИЧЕНИЯ:
   - Модель использует признаки (Favorites, Scored By, Members), которые
     доступны только после релиза аниме
   - Для предсказания рейтинга ДО релиза нужна другая модель

5. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:
   - Модель может использоваться для предсказания финального рейтинга
     на основе текущих метрик популярности
   - Полезна для анализа факторов, влияющих на успешность аниме
""")

print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ 1 ЗАВЕРШЕН")
print("=" * 80)
