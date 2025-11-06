# Prototipo Robo Advisory GenAI

Sistema de recomendaciones personalizadas de productos bancarios utilizando Inteligencia Artificial Generativa.

---

## 📋 Descripción

Este proyecto implementa un **Robo-Advisor Generativo** que combina tres modelos de IA para generar recomendaciones personalizadas de tarjetas de crédito:

1. **Modelo de Análisis de Sentimientos**: Clasifica el sentimiento del cliente (positivo/neutral/negativo)
2. **Red Neuronal de Recomendación**: Predice la probabilidad de que el cliente acepte una nueva tarjeta
3. **GPT-4 para Personalización**: Genera recomendaciones en lenguaje natural adaptadas al perfil del cliente

---

## 🚀 Inicio Rápido

### Requisitos Previos

```bash
Python 3.11+
pip3
```

### Instalación

```bash
# Clonar o descargar el proyecto
cd robo_advisory

# Instalar dependencias
pip3 install pandas numpy scikit-learn tensorflow keras matplotlib seaborn openai
```

### Configuración

```bash
# Configurar API key de OpenAI
export OPENAI_API_KEY="tu-api-key-aqui"
```

### Uso

#### Ejecutar el sistema completo

```bash
python3.11 notebooks/03_robo_advisory_gpt4.py
```

#### Ejecutar modelos individuales

```bash
# Modelo 01: Análisis de Sentimientos
python3.11 notebooks/01_sentiment_analysis.py

# Modelo 02: Red Neuronal de Recomendación
python3.11 notebooks/02_recommendation_neural_network.py

# Modelo 03: Robo Advisory con GPT-4
python3.11 notebooks/03_robo_advisory_gpt4.py
```

---

## 📁 Estructura del Proyecto

```
robo_advisory/
│
├── README.md                       # Este archivo
├── RESUMEN_EJECUTIVO.md            # Resumen del proyecto
│
├── data/                           # Directorio para datos
│
├── models/                         # Modelos entrenados
│   ├── sentiment_model.pkl         # Modelo de sentimientos
│   ├── recommendation_model.h5     # Red neuronal
│   ├── vectorizer.pkl              # Vectorizador de texto
│   ├── label_encoder.pkl           # Codificador de etiquetas
│   ├── scaler.pkl                  # Escalador de features
│   ├── encoders.pkl                # Codificadores categóricos
│   └── feature_names.pkl           # Nombres de features
│
├── notebooks/                      # Scripts de desarrollo
│   ├── 01_sentiment_analysis.py    # Modelo 01
│   ├── 02_recommendation_neural_network.py  # Modelo 02
│   └── 03_robo_advisory_gpt4.py    # Modelo 03 (sistema integrado)
│
├── results/                        # Resultados y métricas
│   ├── sentiment_model_results.csv
│   ├── nn_model_results.csv
│   ├── robo_advisory_results.csv
│   └── personalized_recommendations.txt
│
└── visualizations/                 # Gráficos y visualizaciones
    ├── sentiment_model_comparison.png
    ├── sentiment_confusion_matrix.png
    ├── nn_training_curves.png
    ├── nn_confusion_matrix.png
    ├── nn_roc_curve.png
    └── nn_probability_distribution.png
```

---

## 💻 Uso del Sistema

### Ejemplo de Código

```python
from robo_advisory_system import RoboAdvisorySystem

# Inicializar sistema
robo_advisory = RoboAdvisorySystem(
    sentiment_model_path='models/sentiment_model.pkl',
    nn_model_path='models/recommendation_model.h5',
    vectorizer_path='models/vectorizer.pkl',
    label_encoder_path='models/label_encoder.pkl',
    scaler_path='models/scaler.pkl',
    encoders_path='models/encoders.pkl',
    feature_names_path='models/feature_names.pkl'
)

# Datos del cliente
customer_data = {
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
    'Twitter': 'Great service, very satisfied'
}

# Generar recomendación
result = robo_advisory.generate_personalized_advice(
    customer_id=customer_data['CustomerID'],
    customer_data=customer_data
)

# Mostrar resultado
print(result['personalized_advice'])
```

---

## 📊 Métricas de Performance

### Modelo 01: Análisis de Sentimientos
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1 Score**: 100%

### Modelo 02: Red Neuronal de Recomendación
- **Accuracy**: 98.27%
- **Precision**: 96.98%
- **Recall**: 94.44%
- **F1 Score**: 95.70%
- **AUC**: 99.86%

---

## 🎯 Casos de Uso

### 1. Recomendación Proactiva
Identificar clientes con alta probabilidad de aceptar una nueva tarjeta y enviar recomendaciones personalizadas.

### 2. Atención al Cliente
Asistir a agentes de servicio al cliente con recomendaciones contextualizadas durante interacciones.

### 3. Canales Digitales
Integrar en app móvil o portal web para ofrecer recomendaciones automáticas.

### 4. Recuperación de Clientes
Identificar clientes insatisfechos y ofrecer soluciones personalizadas.

---

## 🔧 Personalización

### Agregar Nuevos Productos

Modificar el Modelo 02 para incluir otros productos:
- Préstamos personales
- Cuentas de ahorro
- Productos de inversión

### Ajustar Criterios de Recomendación

Editar la función `classify_recommendation` en `02_recommendation_neural_network.py`:

```python
def classify_recommendation(row):
    # Personalizar criterios según reglas de negocio
    return (
        (row['Avg_Utilization_Ratio'] > 0.3) & 
        (row['Total_Trans_Ct'] > 50) & 
        (row['Months_on_book'] > 24)
    ).astype(int)
```

### Personalizar Prompts de GPT-4

Editar el prompt en `03_robo_advisory_gpt4.py` para ajustar el tono, formato o contenido de las recomendaciones.

---

## 📈 Visualizaciones Disponibles

### Modelo de Sentimientos
- Comparación de performance entre algoritmos
- Matriz de confusión

### Red Neuronal
- Curvas de entrenamiento (Loss, Accuracy, Precision, Recall)
- Matriz de confusión
- Curva ROC
- Distribución de probabilidades predichas

---

## 🔐 Consideraciones de Seguridad

- **API Keys**: Nunca incluir API keys en el código. Usar variables de entorno.
- **Datos sensibles**: Anonimizar datos de clientes antes de procesamiento.
- **Auditoría**: Mantener logs de todas las recomendaciones generadas.
- **Cumplimiento**: Asegurar cumplimiento con regulaciones bancarias locales.

---

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
# Instalar módulos faltantes
pip3 install <nombre-del-modulo>
```

### Error: "OpenAI API Key not found"
```bash
# Configurar API key
export OPENAI_API_KEY="tu-api-key-aqui"
```

### Error: "Model file not found"
```bash
# Entrenar modelos primero
python3.11 notebooks/01_sentiment_analysis.py
python3.11 notebooks/02_recommendation_neural_network.py
```

---

## 📚 Documentación Adicional

- **Hoja de Ruta Completa**: Ver `/home/ubuntu/hoja_de_ruta_robo_advisory.md`
- **Resumen Ejecutivo**: Ver `RESUMEN_EJECUTIVO.md`
- **Resultados Detallados**: Ver carpeta `results/`

---

## 🤝 Contribuciones

Este es un prototipo de investigación. Para contribuir:

1. Revisar la hoja de ruta completa
2. Identificar áreas de mejora
3. Proponer cambios con justificación técnica
4. Validar con datos reales

---

## 📝 Licencia

Este proyecto es un prototipo de investigación académica.

---

## 📧 Contacto

Para preguntas o colaboraciones, contactar al equipo de investigación.

---

## 🎓 Referencias

- **Transformers**: Vaswani et al. (2017). "Attention is All You Need"
- **Word2Vec**: Mikolov et al. (2013). "Efficient Estimation of Word Representations"
- **Deep Learning**: Goodfellow et al. (2016). "Deep Learning" (MIT Press)
- **OpenAI GPT-4**: OpenAI (2023). "GPT-4 Technical Report"

---

**Última actualización**: Octubre 2025  
**Versión**: 1.0

