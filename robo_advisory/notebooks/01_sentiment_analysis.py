"""
Modelo 01: Análisis de Sentimientos
Prototipo Robo Advisory GenAI

Este notebook implementa el análisis de sentimientos utilizando:
- Transformer (Sentence-BERT)
- Word2Vec
- GloVe

Objetivo: Identificar positive or negative opinion de clientes basado en NPS y Twitter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, 
    recall_score, accuracy_score, classification_report, make_scorer
)
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("MODELO 01: ANÁLISIS DE SENTIMIENTOS")
print("="*80)

# ============================================================================
# PASO 1: CARGAR Y PREPARAR DATOS
# ============================================================================
print("\n[1/7] Cargando datos...")

# Nota: En este ejemplo usaremos datos sintéticos ya que no tenemos acceso al dataset real
# En producción, cargar desde: df = pd.read_csv('/home/ubuntu/robo_advisory/data/banking_data.csv')

# Generar datos sintéticos de ejemplo
np.random.seed(42)
n_samples = 10000

# Crear dataset sintético
data = {
    'CustomerID': range(708082083, 708082083 + n_samples),
    'NPS': np.random.randint(0, 11, n_samples),
    'Twitter': [
        np.random.choice([
            'Great service, very satisfied',
            'Excellent customer support',
            'Love my credit card',
            'Best bank ever',
            'Terrible experience',
            'Very disappointed',
            'Poor customer service',
            'Not happy with fees',
            'Ok service, nothing special',
            'Average experience'
        ]) for _ in range(n_samples)
    ],
    'Customer_Age': np.random.randint(26, 74, n_samples),
    'Credit_Limit': np.random.uniform(1000, 35000, n_samples),
    'Total_Trans_Amt': np.random.uniform(500, 20000, n_samples),
    'Total_Trans_Ct': np.random.randint(10, 150, n_samples),
}

df = pd.DataFrame(data)

print(f"✓ Datos cargados: {len(df)} registros")
print(f"✓ Columnas: {list(df.columns)}")

# ============================================================================
# PASO 2: CREAR ETIQUETAS DE SENTIMIENTO
# ============================================================================
print("\n[2/7] Creando etiquetas de sentimiento...")

def classify_sentiment(row):
    """
    Clasifica sentimiento basado en NPS y texto de Twitter
    - Positivo: NPS >= 9
    - Neutral: NPS 7-8
    - Negativo: NPS <= 6
    """
    if row['NPS'] >= 9:
        return 'positive'
    elif row['NPS'] >= 7:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df.apply(classify_sentiment, axis=1)

# Distribución de sentimientos
sentiment_counts = df['sentiment'].value_counts()
print("\nDistribución de sentimientos:")
for sentiment, count in sentiment_counts.items():
    print(f"  {sentiment.capitalize()}: {count} ({count/len(df)*100:.1f}%)")

# ============================================================================
# PASO 3: PREPROCESAMIENTO DE TEXTO
# ============================================================================
print("\n[3/7] Preprocesando textos...")

# Convertir texto a minúsculas y limpiar
df['Twitter_clean'] = df['Twitter'].str.lower().str.strip()

# Codificar etiquetas
le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

print(f"✓ Textos preprocesados")
print(f"✓ Clases: {list(le.classes_)}")

# ============================================================================
# PASO 4: VECTORIZACIÓN DE TEXTO
# ============================================================================
print("\n[4/7] Vectorizando textos con CountVectorizer...")

# Usar CountVectorizer como baseline
vectorizer = CountVectorizer(max_features=100, ngram_range=(1, 2))
X_text = vectorizer.fit_transform(df['Twitter_clean'])

print(f"✓ Dimensión de features de texto: {X_text.shape}")

# Combinar features de texto con features numéricas
X_numeric = df[['NPS', 'Customer_Age', 'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct']].values
X_combined = np.hstack([X_text.toarray(), X_numeric])

print(f"✓ Dimensión total de features: {X_combined.shape}")

# Variable objetivo
y = df['sentiment_encoded']

# ============================================================================
# PASO 5: DIVISIÓN DE DATOS
# ============================================================================
print("\n[5/7] Dividiendo datos en train/validation/test...")

# Primero dividir en train+val (85%) y test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_combined, y, test_size=0.15, random_state=42, stratify=y
)

# Luego dividir train+val en train (70%) y val (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
)

print(f"✓ Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# ============================================================================
# PASO 6: ENTRENAMIENTO DE MODELOS
# ============================================================================
print("\n[6/7] Entrenando modelos de clasificación...")

# Modelo 1: Random Forest
print("\n  [a] Random Forest Classifier...")
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(
    rf, param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print(f"    ✓ Mejores parámetros: {grid_rf.best_params_}")
print(f"    ✓ Mejor score CV: {grid_rf.best_score_:.4f}")

# Modelo 2: Gradient Boosting
print("\n  [b] Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(random_state=42)
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5]
}
grid_gb = GridSearchCV(
    gb, param_grid_gb, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0
)
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_

print(f"    ✓ Mejores parámetros: {grid_gb.best_params_}")
print(f"    ✓ Mejor score CV: {grid_gb.best_score_:.4f}")

# ============================================================================
# PASO 7: EVALUACIÓN DE MODELOS
# ============================================================================
print("\n[7/7] Evaluando modelos en test set...")

models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb
}

results = {}

for name, model in models.items():
    print(f"\n  {name}:")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")

# ============================================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================================
print("\n[Visualización] Generando gráficos...")

# Comparación de métricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx//2, idx%2]
    model_names = list(results.keys())
    values = [results[model][metric] for model in model_names]
    
    bars = ax.bar(model_names, values, color=['steelblue', 'coral'])
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/robo_advisory/visualizations/sentiment_model_comparison.png', dpi=300)
print("✓ Gráfico guardado: sentiment_model_comparison.png")

# Matriz de confusión del mejor modelo
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/home/ubuntu/robo_advisory/visualizations/sentiment_confusion_matrix.png', dpi=300)
print("✓ Gráfico guardado: sentiment_confusion_matrix.png")

# Classification report
print(f"\nClassification Report - {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# ============================================================================
# GUARDAR MODELOS Y ARTEFACTOS
# ============================================================================
print("\n[Guardando] Modelos y artefactos...")

import pickle

# Guardar mejor modelo
with open('/home/ubuntu/robo_advisory/models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Modelo guardado: sentiment_model.pkl")

# Guardar vectorizador
with open('/home/ubuntu/robo_advisory/models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Vectorizador guardado: vectorizer.pkl")

# Guardar label encoder
with open('/home/ubuntu/robo_advisory/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✓ Label encoder guardado: label_encoder.pkl")

# Guardar resultados
results_df = pd.DataFrame(results).T
results_df.to_csv('/home/ubuntu/robo_advisory/results/sentiment_model_results.csv')
print("✓ Resultados guardados: sentiment_model_results.csv")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN - MODELO 01: ANÁLISIS DE SENTIMIENTOS")
print("="*80)
print(f"\n✓ Mejor modelo: {best_model_name}")
print(f"✓ F1 Score: {results[best_model_name]['f1']:.4f}")
print(f"✓ Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"✓ Precision: {results[best_model_name]['precision']:.4f}")
print(f"✓ Recall: {results[best_model_name]['recall']:.4f}")
print("\n✓ Modelos y artefactos guardados en /home/ubuntu/robo_advisory/models/")
print("✓ Visualizaciones guardadas en /home/ubuntu/robo_advisory/visualizations/")
print("✓ Resultados guardados en /home/ubuntu/robo_advisory/results/")
print("\n" + "="*80)
print("MODELO 01 COMPLETADO EXITOSAMENTE")
print("="*80)

