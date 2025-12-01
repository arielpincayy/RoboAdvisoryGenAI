#!/usr/bin/env python3
"""
Script para crear el pipeline de preprocesamiento de datos de clientes.
DEBE ejecutarse FUERA del Docker, en tu máquina local.
"""

import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

# ⚠️ CRÍTICO: Verificar que data_transformers.py existe
if not os.path.exists('data_transformers.py'):
    print("❌ ERROR: No se encuentra data_transformers.py en el directorio actual")
    print("   Asegúrate de que data_transformers.py esté en el mismo directorio")
    sys.exit(1)

# Importar desde el módulo externo
from data_transformers import DropColumns, DynamicPreprocessor, DateFeatureGenerator

# --- Directorios ---
PATH_PIPELINE = 'pipelines'
PATH_DATA = 'data'

os.makedirs(PATH_PIPELINE, exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)

# --- Cargar datos ---
print("="*60)
print("🚀 CREANDO PIPELINE DE DATOS DE CLIENTES")
print("="*60)

CSV_PATH = os.path.join(PATH_DATA, 'BankChurners_merged.csv')
if not os.path.exists(CSV_PATH):
    print(f"❌ ERROR: No se encuentra {CSV_PATH}")
    print("   Asegúrate de que el archivo CSV exista en data/")
    sys.exit(1)

print(f"\n📂 Cargando datos desde: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(f"   • Filas originales: {len(df)}")
print(f"   • Columnas: {len(df.columns)}")

# Eliminar filas con valores nulos
df.dropna(inplace=True)
print(f"   • Filas después de dropna: {len(df)}")

# --- Crear pipeline ---
print("\n🔧 Creando pipeline...")
print(f"   • DropColumns importado desde: {DropColumns.__module__}")
print(f"   • DynamicPreprocessor importado desde: {DynamicPreprocessor.__module__}")
print(f"   • DateFeatureGenerator importado desde: {DateFeatureGenerator.__module__}")

data_pipeline = Pipeline([
    ('date_feature', DateFeatureGenerator()),
    ('drop_cols', DropColumns()),
    ('preprocessor', DynamicPreprocessor())
])

# --- Entrenar ---
print("\n🏋️  Entrenando pipeline...")
data_pipeline.fit(df)
print("✅ Pipeline entrenado")

# Mostrar información del preprocessing
preprocessor = data_pipeline.named_steps['preprocessor']
print(f"\n📊 Información del preprocessing:")
print(f"   • Columnas numéricas: {len(preprocessor.num_cols)}")
print(f"   • Columnas categóricas: {len(preprocessor.cat_cols)}")
print(f"   • Features generadas: {len(preprocessor.num_cols) + len(preprocessor.cat_feature_names)}")

# --- Guardar ---
PIPELINE_OUTPUT = os.path.join(PATH_PIPELINE, 'pipeline_bankchurner_preprocessing.joblib')

print(f"\n💾 Guardando pipeline en: {PIPELINE_OUTPUT}")
joblib.dump(data_pipeline, PIPELINE_OUTPUT)
print("✅ Pipeline guardado")

# --- Verificar ---
print("\n🔍 Verificando carga del pipeline...")
loaded_pipeline = joblib.load(PIPELINE_OUTPUT)

# Verificar módulos
date_feature = loaded_pipeline.named_steps['date_feature']
drop_cols = loaded_pipeline.named_steps['drop_cols']
preprocessor = loaded_pipeline.named_steps['preprocessor']

print(f"   • DateFeatureGenerator:")
print(f"     - Clase: {date_feature.__class__.__name__}")
print(f"     - Módulo: {date_feature.__class__.__module__}")

print(f"   • DropColumns:")
print(f"     - Clase: {drop_cols.__class__.__name__}")
print(f"     - Módulo: {drop_cols.__class__.__module__}")

print(f"   • DynamicPreprocessor:")
print(f"     - Clase: {preprocessor.__class__.__name__}")
print(f"     - Módulo: {preprocessor.__class__.__module__}")

# Verificar que están correctamente vinculados
if (date_feature.__class__.__module__ == 'data_transformers' and
    drop_cols.__class__.__module__ == 'data_transformers' and 
    preprocessor.__class__.__module__ == 'data_transformers'):
    print("\n   ✅ Pipeline correctamente vinculado a data_transformers.py")
else:
    print("\n   ⚠️  ADVERTENCIA: Algún transformador no está vinculado correctamente")

# --- Test ---
print("\n🧪 Probando transformación...")
test_df = df.head(5)
result = loaded_pipeline.transform(test_df)

print(f"   ✓ Input shape: {test_df.shape}")
print(f"   ✓ Output shape: {result.shape}")
print(f"   ✓ Output columns: {len(result.columns)}")

# --- Resumen ---
print("\n" + "="*60)
print("🎉 ¡PIPELINE DE DATOS CREADO EXITOSAMENTE!")
print("="*60)
print(f"\n📁 Archivo generado:")
print(f"   • {PIPELINE_OUTPUT}")

print("\n📦 Para usar en Docker:")
print("   1. Asegúrate de copiar data_transformers.py al contenedor")
print("   2. Copia la carpeta pipelines/ completa")
print("\n💡 Ahora puedes construir tu imagen Docker")