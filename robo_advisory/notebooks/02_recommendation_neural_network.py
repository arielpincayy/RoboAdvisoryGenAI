"""
Modelo 02: Red Neuronal de Recomendación
Prototipo Robo Advisory GenAI

Este notebook implementa una red neuronal para predecir recomendaciones de tarjeta de crédito.

Arquitectura:
- Funciones de activación: ReLU y Sigmoid
- Optimizador: SGD (equivalente a XSGD)
- Capas: Dense con Dropout para regularización

Objetivo: Predecir si recomendar tarjeta de crédito a un cliente (clasificación binaria)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("MODELO 02: RED NEURONAL DE RECOMENDACIÓN")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# ============================================================================
# PASO 1: CARGAR Y PREPARAR DATOS
# ============================================================================
print("\n[1/8] Cargando datos...")

# Generar datos sintéticos de ejemplo
np.random.seed(42)
n_samples = 10000

# Crear dataset sintético más realista
data = {
    'CustomerID': range(708082083, 708082083 + n_samples),
    'Customer_Age': np.random.randint(26, 74, n_samples),
    'Gender': np.random.choice(['M', 'F'], n_samples),
    'Education_Level': np.random.choice(['High School', 'Graduate', 'Post-Graduate', 'Doctorate'], n_samples),
    'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
    'Income_Category': np.random.choice(['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'], n_samples),
    'Credit_Limit': np.random.uniform(1000, 35000, n_samples),
    'Total_Revolving_Bal': np.random.uniform(0, 2500, n_samples),
    'Avg_Utilization_Ratio': np.random.uniform(0, 1, n_samples),
    'Total_Trans_Amt': np.random.uniform(500, 20000, n_samples),
    'Total_Trans_Ct': np.random.randint(10, 150, n_samples),
    'Total_Relationship_Count': np.random.randint(1, 7, n_samples),
    'Months_on_book': np.random.randint(12, 60, n_samples),
    'Contacts_Count_12_mon': np.random.randint(0, 7, n_samples),
}

df = pd.DataFrame(data)

# Crear variable objetivo: ¿Recomendar tarjeta de crédito?
# Criterios: alta utilización, muchas transacciones, buen historial
df['recommend_credit_card'] = (
    (df['Avg_Utilization_Ratio'] > 0.3) & 
    (df['Total_Trans_Ct'] > 50) & 
    (df['Months_on_book'] > 24) &
    (df['Credit_Limit'] < 20000)
).astype(int)

print(f"✓ Datos cargados: {len(df)} registros")
print(f"✓ Columnas: {list(df.columns)}")

# Distribución de la variable objetivo
target_counts = df['recommend_credit_card'].value_counts()
print(f"\nDistribución de la variable objetivo:")
print(f"  No recomendar (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"  Recomendar (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")

# ============================================================================
# PASO 2: INGENIERÍA DE CARACTERÍSTICAS
# ============================================================================
print("\n[2/8] Realizando ingeniería de características...")

# Separar features numéricas y categóricas
numerical_features = [
    'Customer_Age', 'Credit_Limit', 'Total_Revolving_Bal', 
    'Avg_Utilization_Ratio', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Relationship_Count', 'Months_on_book', 'Contacts_Count_12_mon'
]

categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category']

print(f"✓ Features numéricas: {len(numerical_features)}")
print(f"✓ Features categóricas: {len(categorical_features)}")

# ============================================================================
# PASO 3: PREPROCESAMIENTO
# ============================================================================
print("\n[3/8] Preprocesando datos...")

# Codificar variables categóricas
encoders = {}
encoded_features = []

for feature in categorical_features:
    le = LabelEncoder()
    df[f'{feature}_encoded'] = le.fit_transform(df[feature])
    encoders[feature] = le
    encoded_features.append(f'{feature}_encoded')

print(f"✓ Variables categóricas codificadas")

# Preparar matriz de features
X = df[numerical_features + encoded_features].values
y = df['recommend_credit_card'].values

print(f"✓ Dimensión de X: {X.shape}")
print(f"✓ Dimensión de y: {y.shape}")

# Normalizar features numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Features normalizadas")

# ============================================================================
# PASO 4: DIVISIÓN DE DATOS
# ============================================================================
print("\n[4/8] Dividiendo datos en train/validation/test...")

# Primero dividir en train+val (85%) y test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)

# Luego dividir train+val en train (70%) y val (15%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
)

print(f"✓ Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"✓ Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# ============================================================================
# PASO 5: CONSTRUCCIÓN DE LA RED NEURONAL
# ============================================================================
print("\n[5/8] Construyendo arquitectura de la red neuronal...")

n_features = X_train.shape[1]

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(n_features,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))  # Clasificación binaria

# Compilar modelo con optimizador SGD (equivalente a XSGD)
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

print("\n✓ Arquitectura del modelo:")
model.summary()

# ============================================================================
# PASO 6: ENTRENAMIENTO DEL MODELO
# ============================================================================
print("\n[6/8] Entrenando la red neuronal...")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    '/home/ubuntu/robo_advisory/models/recommendation_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Entrenar
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\n✓ Entrenamiento completado")

# ============================================================================
# PASO 7: EVALUACIÓN DEL MODELO
# ============================================================================
print("\n[7/8] Evaluando el modelo en test set...")

# Evaluar en test set
test_results = model.evaluate(X_test, y_test, verbose=0)
test_loss, test_accuracy, test_precision, test_recall, test_auc = test_results

print(f"\nMétricas en Test Set:")
print(f"  Loss:      {test_loss:.4f}")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  AUC:       {test_auc:.4f}")

# Calcular F1 Score
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
print(f"  F1 Score:  {f1_score:.4f}")

# Predicciones
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de Confusión:")
print(cm)

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Recomendar', 'Recomendar']))

# ============================================================================
# PASO 8: VISUALIZACIÓN DE RESULTADOS
# ============================================================================
print("\n[8/8] Generando visualizaciones...")

# 1. Curvas de entrenamiento
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Curves')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision Curves')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_title('Recall Curves')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/robo_advisory/visualizations/nn_training_curves.png', dpi=300)
print("✓ Gráfico guardado: nn_training_curves.png")

# 2. Matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Recomendar', 'Recomendar'],
            yticklabels=['No Recomendar', 'Recomendar'])
plt.title('Confusion Matrix - Neural Network')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('/home/ubuntu/robo_advisory/visualizations/nn_confusion_matrix.png', dpi=300)
print("✓ Gráfico guardado: nn_confusion_matrix.png")

# 3. Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/robo_advisory/visualizations/nn_roc_curve.png', dpi=300)
print("✓ Gráfico guardado: nn_roc_curve.png")

# 4. Distribución de probabilidades predichas
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='No Recomendar (True)', color='blue')
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Recomendar (True)', color='red')
plt.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold = 0.5')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/robo_advisory/visualizations/nn_probability_distribution.png', dpi=300)
print("✓ Gráfico guardado: nn_probability_distribution.png")

# ============================================================================
# GUARDAR MODELOS Y ARTEFACTOS
# ============================================================================
print("\n[Guardando] Modelos y artefactos...")

import pickle

# Guardar scaler
with open('/home/ubuntu/robo_advisory/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler guardado: scaler.pkl")

# Guardar encoders
with open('/home/ubuntu/robo_advisory/models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print("✓ Encoders guardados: encoders.pkl")

# Guardar nombres de features
feature_names = numerical_features + encoded_features
with open('/home/ubuntu/robo_advisory/models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Feature names guardados: feature_names.pkl")

# Guardar resultados
results = {
    'loss': test_loss,
    'accuracy': test_accuracy,
    'precision': test_precision,
    'recall': test_recall,
    'f1_score': f1_score,
    'auc': test_auc
}

results_df = pd.DataFrame([results])
results_df.to_csv('/home/ubuntu/robo_advisory/results/nn_model_results.csv', index=False)
print("✓ Resultados guardados: nn_model_results.csv")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN - MODELO 02: RED NEURONAL DE RECOMENDACIÓN")
print("="*80)
print(f"\n✓ Arquitectura: Sequential con 4 capas Dense + 3 Dropout")
print(f"✓ Optimizador: SGD (learning_rate=0.01, momentum=0.9)")
print(f"✓ Función de activación: ReLU (capas ocultas), Sigmoid (salida)")
print(f"\nMétricas en Test Set:")
print(f"  • Accuracy:  {test_accuracy:.4f}")
print(f"  • Precision: {test_precision:.4f}")
print(f"  • Recall:    {test_recall:.4f}")
print(f"  • F1 Score:  {f1_score:.4f}")
print(f"  • AUC:       {test_auc:.4f}")
print("\n✓ Modelo guardado en /home/ubuntu/robo_advisory/models/recommendation_model.h5")
print("✓ Visualizaciones guardadas en /home/ubuntu/robo_advisory/visualizations/")
print("✓ Resultados guardados en /home/ubuntu/robo_advisory/results/")
print("\n" + "="*80)
print("MODELO 02 COMPLETADO EXITOSAMENTE")
print("="*80)

