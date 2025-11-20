import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
from text_processing import TextPreprocessor

# --- Rutas ---
PATH_PIPELINE = 'pipelines'
PATH_DATA = 'data'
PATH_MODELS = 'models'

os.makedirs(PATH_PIPELINE, exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_MODELS, exist_ok=True)

# --- Dataset ---
print("📂 Cargando datos...")
df_text = pd.read_csv(os.path.join(PATH_DATA, "tweets_corregidos.csv"))
df_text.dropna(subset=['Twitter'], inplace=True)
X_text = df_text['Twitter']
print(f"✅ Cargadas {len(X_text)} muestras")

# ----------------------------
# MAIN: Crear y guardar pipeline
# ----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INICIANDO CREACIÓN DE PIPELINE CON JOBLIB")
    print("="*60 + "\n")
    
    # Configurar rutas
    W2V_PATH = os.path.join(PATH_PIPELINE, 'word2vec_model.bin')
    GLOVE_PATH = os.path.join(PATH_MODELS, 'glove.6B.100d.txt')
    PIPELINE_OUTPUT = os.path.join(PATH_PIPELINE, 'text_pipeline.joblib')
    
    # Verificar si existe GloVe (opcional)
    if not os.path.exists(GLOVE_PATH):
        print(f"⚠️  No se encontró GloVe en: {GLOVE_PATH}")
        print("   El pipeline se creará sin embeddings GloVe")
        GLOVE_PATH = None
    
    # Crear pipeline
    text_pipeline = Pipeline([
        ('text_proc', TextPreprocessor(
            use_bigrams=True,
            use_trigrams=True,
            glove_path=GLOVE_PATH,
            sbert_model_name='all-MiniLM-L6-v2',
            w2v_model_path=W2V_PATH
        ))
    ])

    # Entrenar pipeline
    print("\n🏋️  Entrenando pipeline...")
    text_pipeline.fit(X_text)

    # Limpiar modelos de memoria antes de guardar
    print("\n🧹 Limpiando modelos de memoria antes de serializar...")
    text_pipeline.named_steps['text_proc'].w2v_model = None
    text_pipeline.named_steps['text_proc'].sbert_model = None
    text_pipeline.named_steps['text_proc'].stop_words = None
    text_pipeline.named_steps['text_proc'].lemmatizer = None

    # Guardar con joblib
    print(f"\n💾 Guardando pipeline en: {PIPELINE_OUTPUT}")
    joblib.dump(text_pipeline, PIPELINE_OUTPUT)
    
    print("\n✅ Pipeline de texto guardado exitosamente con joblib")
    
    # Verificar que se puede cargar
    print("\n🔍 Verificando carga del pipeline...")
    loaded_pipeline = joblib.load(PIPELINE_OUTPUT)
    
    print("✅ Pipeline cargado correctamente!")
    
    # Test rápido
    print("\n🧪 Probando transformación...")
    test_text = pd.Series(["This is a great credit card service!"])
    result = loaded_pipeline.transform(test_text)
    
    print(f"   ✓ W2V shape: {result['w2v'].shape}")
    print(f"   ✓ SBERT shape: {result['sbert'].shape}")
    print(f"   ✓ GloVe shape: {result['glove'].shape if result['glove'] is not None else 'None (no usado)'}")
    
    print("\n" + "="*60)
    print("🎉 ¡PIPELINE CREADO Y VERIFICADO EXITOSAMENTE!")
    print("="*60)
    print(f"\n📁 Archivos generados:")
    print(f"   • Pipeline: {PIPELINE_OUTPUT}")
    print(f"   • Word2Vec: {W2V_PATH}")
    if GLOVE_PATH:
        print(f"   • GloVe: {GLOVE_PATH} (referenciado)")
    print("\n💡 Ahora puedes copiar estos archivos a tu Docker")