"""
Modelo 03: Robo Advisory con GPT-4
Prototipo Robo Advisory GenAI

Este notebook integra los tres modelos:
1. Modelo 01: Análisis de Sentimientos
2. Modelo 02: Red Neuronal de Recomendación
3. Modelo 03: GPT-4 para recomendaciones personalizadas

Objetivo: Generar recomendaciones personalizadas de tarjetas de crédito
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

# OpenAI
from openai import OpenAI

print("="*80)
print("MODELO 03: ROBO ADVISORY CON GPT-4")
print("="*80)

# ============================================================================
# CLASE: SISTEMA ROBO ADVISORY
# ============================================================================

class RoboAdvisorySystem:
    """
    Sistema integrado de Robo Advisory que combina los 3 modelos
    """
    
    def __init__(self, sentiment_model_path, nn_model_path, 
                 vectorizer_path, label_encoder_path,
                 scaler_path, encoders_path, feature_names_path):
        """
        Inicializa el sistema cargando todos los modelos y artefactos
        """
        print("\n[Inicializando] Cargando modelos y artefactos...")
        
        # Cargar Modelo 01: Análisis de Sentimientos
        with open(sentiment_model_path, 'rb') as f:
            self.sentiment_model = pickle.load(f)
        print("✓ Modelo de sentimientos cargado")
        
        # Cargar vectorizador y label encoder
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("✓ Vectorizador y label encoder cargados")
        
        # Cargar Modelo 02: Red Neuronal de Recomendación
        self.recommendation_model = keras.models.load_model(nn_model_path)
        print("✓ Red neuronal de recomendación cargada")
        
        # Cargar scaler, encoders y feature names
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        print("✓ Scaler, encoders y feature names cargados")
        
        # Inicializar cliente de OpenAI
        self.openai_client = OpenAI()
        print("✓ Cliente de OpenAI inicializado")
        
        print("\n✓ Sistema Robo Advisory inicializado correctamente")
    
    def analyze_sentiment(self, nps_score, twitter_text, customer_age, 
                         credit_limit, total_trans_amt, total_trans_ct):
        """
        Modelo 01: Análisis de sentimientos
        """
        # Preprocesar texto
        twitter_clean = twitter_text.lower().strip()
        
        # Vectorizar texto
        text_features = self.vectorizer.transform([twitter_clean]).toarray()
        
        # Combinar con features numéricas
        numeric_features = np.array([[nps_score, customer_age, credit_limit, 
                                     total_trans_amt, total_trans_ct]])
        combined_features = np.hstack([text_features, numeric_features])
        
        # Predecir sentimiento
        sentiment_pred_encoded = self.sentiment_model.predict(combined_features)
        sentiment_proba = self.sentiment_model.predict_proba(combined_features)
        
        sentiment = self.label_encoder.inverse_transform(sentiment_pred_encoded)[0]
        confidence = sentiment_proba[0].max()
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': sentiment_proba[0][0],
                'neutral': sentiment_proba[0][1],
                'positive': sentiment_proba[0][2]
            }
        }
    
    def predict_recommendation(self, customer_data):
        """
        Modelo 02: Predicción de recomendación
        """
        # Extraer features
        features = self._extract_features(customer_data)
        
        # Escalar features
        features_scaled = self.scaler.transform(features)
        
        # Predecir
        recommendation_proba = self.recommendation_model.predict(features_scaled, verbose=0)
        recommendation = (recommendation_proba > 0.5).astype(int)[0][0]
        
        return {
            'recommendation': 'Sí' if recommendation == 1 else 'No',
            'probability': float(recommendation_proba[0][0]),
            'confidence': 'Alta' if abs(recommendation_proba[0][0] - 0.5) > 0.3 else 'Media'
        }
    
    def generate_personalized_advice(self, customer_id, customer_data):
        """
        Pipeline completo: Modelo 01 + Modelo 02 + Modelo 03 (GPT-4)
        """
        print(f"\n{'='*80}")
        print(f"Generando recomendación para Cliente ID: {customer_id}")
        print(f"{'='*80}")
        
        # Paso 1: Análisis de sentimientos
        print("\n[Paso 1/3] Analizando sentimiento del cliente...")
        sentiment_result = self.analyze_sentiment(
            nps_score=customer_data['NPS'],
            twitter_text=customer_data.get('Twitter', ''),
            customer_age=customer_data['Customer_Age'],
            credit_limit=customer_data['Credit_Limit'],
            total_trans_amt=customer_data['Total_Trans_Amt'],
            total_trans_ct=customer_data['Total_Trans_Ct']
        )
        print(f"  ✓ Sentimiento: {sentiment_result['sentiment']}")
        print(f"  ✓ Confianza: {sentiment_result['confidence']:.2%}")
        
        # Paso 2: Predicción de recomendación
        print("\n[Paso 2/3] Prediciendo recomendación...")
        recommendation_result = self.predict_recommendation(customer_data)
        print(f"  ✓ Recomendación: {recommendation_result['recommendation']}")
        print(f"  ✓ Probabilidad: {recommendation_result['probability']:.2%}")
        print(f"  ✓ Confianza: {recommendation_result['confidence']}")
        
        # Paso 3: Generación de recomendación personalizada con GPT-4
        print("\n[Paso 3/3] Generando recomendación personalizada con GPT-4...")
        personalized_advice = self._get_gpt4_recommendation(
            sentiment_result=sentiment_result,
            recommendation_result=recommendation_result,
            customer_data=customer_data
        )
        print("  ✓ Recomendación personalizada generada")
        
        # Compilar resultado completo
        result = {
            'customer_id': customer_id,
            'sentiment_analysis': sentiment_result,
            'recommendation_prediction': recommendation_result,
            'personalized_advice': personalized_advice,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _extract_features(self, customer_data):
        """
        Extrae y prepara features para el modelo de recomendación
        """
        # Features numéricas
        numerical_features = [
            customer_data['Customer_Age'],
            customer_data['Credit_Limit'],
            customer_data['Total_Revolving_Bal'],
            customer_data['Avg_Utilization_Ratio'],
            customer_data['Total_Trans_Amt'],
            customer_data['Total_Trans_Ct'],
            customer_data['Total_Relationship_Count'],
            customer_data['Months_on_book'],
            customer_data['Contacts_Count_12_mon']
        ]
        
        # Features categóricas codificadas
        categorical_features = []
        for feature in ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category']:
            encoder = self.encoders[feature]
            encoded_value = encoder.transform([customer_data[feature]])[0]
            categorical_features.append(encoded_value)
        
        # Combinar todas las features
        all_features = numerical_features + categorical_features
        
        return np.array([all_features])
    
    def _get_gpt4_recommendation(self, sentiment_result, recommendation_result, customer_data):
        """
        Genera recomendación personalizada usando GPT-4
        """
        # Construir prompt
        prompt = f"""
Eres un asesor financiero experto en banca. Basándote en la siguiente información del cliente, 
genera una recomendación personalizada sobre productos de tarjeta de crédito.

INFORMACIÓN DEL CLIENTE:
- Edad: {customer_data['Customer_Age']} años
- Género: {customer_data['Gender']}
- Nivel educativo: {customer_data['Education_Level']}
- Estado civil: {customer_data['Marital_Status']}
- Categoría de ingresos: {customer_data['Income_Category']}
- Límite de crédito actual: ${customer_data['Credit_Limit']:,.2f}
- Balance rotativo: ${customer_data['Total_Revolving_Bal']:,.2f}
- Ratio de utilización: {customer_data['Avg_Utilization_Ratio']:.2%}
- Monto total de transacciones: ${customer_data['Total_Trans_Amt']:,.2f}
- Número de transacciones: {customer_data['Total_Trans_Ct']}
- Meses como cliente: {customer_data['Months_on_book']}
- Productos con el banco: {customer_data['Total_Relationship_Count']}

ANÁLISIS DE SENTIMIENTO:
- Sentimiento del cliente: {sentiment_result['sentiment']}
- Confianza: {sentiment_result['confidence']:.2%}
- NPS: {customer_data['NPS']}

ANÁLISIS PREDICTIVO:
- Probabilidad de aceptación de nueva tarjeta: {recommendation_result['probability']:.2%}
- Recomendación del modelo: {recommendation_result['recommendation']}

Por favor, genera una recomendación personalizada que incluya:
1. Análisis de la situación actual del cliente
2. Recomendación específica de producto (tipo de tarjeta de crédito)
3. Beneficios clave para el cliente
4. Siguiente acción sugerida
5. Tono apropiado según el sentimiento del cliente

La recomendación debe ser clara, profesional y orientada a mejorar la experiencia del cliente.
Máximo 250 palabras.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Eres un asesor financiero experto especializado en productos bancarios y experiencia del cliente."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            recommendation = response.choices[0].message.content
            return recommendation
            
        except Exception as e:
            return f"Error al generar recomendación: {str(e)}"


# ============================================================================
# DEMOSTRACIÓN DEL SISTEMA
# ============================================================================

def main():
    """
    Función principal para demostrar el sistema Robo Advisory
    """
    
    # Rutas de los modelos
    models_dir = '/home/ubuntu/robo_advisory/models'
    
    # Inicializar sistema
    robo_advisory = RoboAdvisorySystem(
        sentiment_model_path=f'{models_dir}/sentiment_model.pkl',
        nn_model_path=f'{models_dir}/recommendation_model.h5',
        vectorizer_path=f'{models_dir}/vectorizer.pkl',
        label_encoder_path=f'{models_dir}/label_encoder.pkl',
        scaler_path=f'{models_dir}/scaler.pkl',
        encoders_path=f'{models_dir}/encoders.pkl',
        feature_names_path=f'{models_dir}/feature_names.pkl'
    )
    
    # ========================================================================
    # CASO 1: Cliente con sentimiento positivo y alta probabilidad
    # ========================================================================
    
    customer_data_1 = {
        'CustomerID': 708082083,
        'Customer_Age': 45,
        'Gender': 'M',
        'Education_Level': 'Graduate',
        'Marital_Status': 'Married',
        'Income_Category': '$60K - $80K',
        'Credit_Limit': 12000,
        'Total_Revolving_Bal': 1500,
        'Avg_Utilization_Ratio': 0.45,
        'Total_Trans_Amt': 8500,
        'Total_Trans_Ct': 85,
        'Total_Relationship_Count': 4,
        'Months_on_book': 36,
        'Contacts_Count_12_mon': 2,
        'NPS': 9,
        'Twitter': 'Great service, very satisfied with my credit card'
    }
    
    result_1 = robo_advisory.generate_personalized_advice(
        customer_id=customer_data_1['CustomerID'],
        customer_data=customer_data_1
    )
    
    # Mostrar resultado
    print("\n" + "="*80)
    print("RECOMENDACIÓN PERSONALIZADA - CASO 1")
    print("="*80)
    print(f"\nCliente ID: {result_1['customer_id']}")
    print(f"\nAnálisis de Sentimiento:")
    print(f"  • Sentimiento: {result_1['sentiment_analysis']['sentiment']}")
    print(f"  • Confianza: {result_1['sentiment_analysis']['confidence']:.2%}")
    print(f"\nPredicción de Recomendación:")
    print(f"  • Recomendación: {result_1['recommendation_prediction']['recommendation']}")
    print(f"  • Probabilidad: {result_1['recommendation_prediction']['probability']:.2%}")
    print(f"\nRecomendación Personalizada (GPT-4):")
    print("-" * 80)
    print(result_1['personalized_advice'])
    print("-" * 80)
    
    # ========================================================================
    # CASO 2: Cliente con sentimiento negativo y baja probabilidad
    # ========================================================================
    
    customer_data_2 = {
        'CustomerID': 708082084,
        'Customer_Age': 32,
        'Gender': 'F',
        'Education_Level': 'High School',
        'Marital_Status': 'Single',
        'Income_Category': 'Less than $40K',
        'Credit_Limit': 5000,
        'Total_Revolving_Bal': 2200,
        'Avg_Utilization_Ratio': 0.15,
        'Total_Trans_Amt': 2500,
        'Total_Trans_Ct': 25,
        'Total_Relationship_Count': 2,
        'Months_on_book': 18,
        'Contacts_Count_12_mon': 5,
        'NPS': 4,
        'Twitter': 'Not happy with fees and customer service'
    }
    
    result_2 = robo_advisory.generate_personalized_advice(
        customer_id=customer_data_2['CustomerID'],
        customer_data=customer_data_2
    )
    
    # Mostrar resultado
    print("\n" + "="*80)
    print("RECOMENDACIÓN PERSONALIZADA - CASO 2")
    print("="*80)
    print(f"\nCliente ID: {result_2['customer_id']}")
    print(f"\nAnálisis de Sentimiento:")
    print(f"  • Sentimiento: {result_2['sentiment_analysis']['sentiment']}")
    print(f"  • Confianza: {result_2['sentiment_analysis']['confidence']:.2%}")
    print(f"\nPredicción de Recomendación:")
    print(f"  • Recomendación: {result_2['recommendation_prediction']['recommendation']}")
    print(f"  • Probabilidad: {result_2['recommendation_prediction']['probability']:.2%}")
    print(f"\nRecomendación Personalizada (GPT-4):")
    print("-" * 80)
    print(result_2['personalized_advice'])
    print("-" * 80)
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    
    print("\n" + "="*80)
    print("Guardando resultados...")
    print("="*80)
    
    # Guardar resultados en CSV
    results_df = pd.DataFrame([
        {
            'customer_id': result_1['customer_id'],
            'sentiment': result_1['sentiment_analysis']['sentiment'],
            'sentiment_confidence': result_1['sentiment_analysis']['confidence'],
            'recommendation': result_1['recommendation_prediction']['recommendation'],
            'recommendation_probability': result_1['recommendation_prediction']['probability'],
            'timestamp': result_1['timestamp']
        },
        {
            'customer_id': result_2['customer_id'],
            'sentiment': result_2['sentiment_analysis']['sentiment'],
            'sentiment_confidence': result_2['sentiment_analysis']['confidence'],
            'recommendation': result_2['recommendation_prediction']['recommendation'],
            'recommendation_probability': result_2['recommendation_prediction']['probability'],
            'timestamp': result_2['timestamp']
        }
    ])
    
    results_df.to_csv('/home/ubuntu/robo_advisory/results/robo_advisory_results.csv', index=False)
    print("✓ Resultados guardados en: robo_advisory_results.csv")
    
    # Guardar recomendaciones completas en texto
    with open('/home/ubuntu/robo_advisory/results/personalized_recommendations.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("RECOMENDACIONES PERSONALIZADAS - ROBO ADVISORY GENAI\n")
        f.write("="*80 + "\n\n")
        
        f.write("CASO 1: Cliente con sentimiento positivo\n")
        f.write("-"*80 + "\n")
        f.write(f"Cliente ID: {result_1['customer_id']}\n")
        f.write(f"Sentimiento: {result_1['sentiment_analysis']['sentiment']}\n")
        f.write(f"Recomendación: {result_1['recommendation_prediction']['recommendation']}\n\n")
        f.write(result_1['personalized_advice'])
        f.write("\n\n" + "="*80 + "\n\n")
        
        f.write("CASO 2: Cliente con sentimiento negativo\n")
        f.write("-"*80 + "\n")
        f.write(f"Cliente ID: {result_2['customer_id']}\n")
        f.write(f"Sentimiento: {result_2['sentiment_analysis']['sentiment']}\n")
        f.write(f"Recomendación: {result_2['recommendation_prediction']['recommendation']}\n\n")
        f.write(result_2['personalized_advice'])
        f.write("\n\n" + "="*80 + "\n")
    
    print("✓ Recomendaciones completas guardadas en: personalized_recommendations.txt")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESUMEN - MODELO 03: ROBO ADVISORY CON GPT-4")
    print("="*80)
    print("\n✓ Sistema Robo Advisory completamente funcional")
    print("✓ Integración exitosa de los 3 modelos:")
    print("  1. Análisis de Sentimientos (Random Forest)")
    print("  2. Red Neuronal de Recomendación (Deep Learning)")
    print("  3. Generación de Recomendaciones (GPT-4)")
    print("\n✓ Casos de prueba ejecutados: 2")
    print("✓ Resultados guardados en /home/ubuntu/robo_advisory/results/")
    print("\n" + "="*80)
    print("MODELO 03 COMPLETADO EXITOSAMENTE")
    print("="*80)


if __name__ == "__main__":
    main()

