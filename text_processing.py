import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer



class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, use_bigrams=True, use_trigrams=True, glove_path=None,
                 sbert_model_name='all-MiniLM-L6-v2', w2v_model_path=None):
        self.use_bigrams = use_bigrams
        self.use_trigrams = use_trigrams
        self.glove_path = glove_path
        self.glove = {}
        self.sbert_model_name = sbert_model_name
        self.sbert_model = None
        self.w2v_model = None
        self.w2v_model_path = w2v_model_path

        # Inicializar recursos NLTK
        self._ensure_nltk_resources()
        self._init_nltk_components()

    # ----------------------------
    # Métodos de inicialización
    # ----------------------------
    def _ensure_nltk_resources(self):
        """Descarga recursos NLTK si no están disponibles"""
        import nltk
        
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif resource == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif resource == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                elif resource == 'omw-1.4':
                    nltk.data.find('corpora/omw-1.4')
            except LookupError:
                print(f"📥 Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
        
        # ⚠️ CRÍTICO: Descargar tagger con fallback
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            try:
                print("📥 Descargando averaged_perceptron_tagger_eng...")
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            except:
                print("⚠️ Fallback: Descargando averaged_perceptron_tagger...")
                nltk.download('averaged_perceptron_tagger', quiet=True)

    def _init_nltk_components(self):
        """Inicializa componentes de NLTK"""
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # ----------------------------
    # Utilidades de procesamiento
    # ----------------------------
    def _clean_text(self, text):
        """Limpia y normaliza texto"""
        import re
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-záéíóúüñ ]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_wordnet_pos(self, tag):
        """Convierte tag POS a formato WordNet"""
        from nltk.corpus import wordnet
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _tokenize_series(self, series):
        """Tokeniza, lematiza y genera n-gramas"""
        from nltk import pos_tag, ngrams
        from nltk.tokenize import word_tokenize
        
        all_tokens = []
        total = len(series)
        
        for idx, text in enumerate(series):
            if (idx + 1) % 500 == 0:
                print(f"   Procesando: {idx + 1}/{total} textos...")
            
            text_clean = self._clean_text(text)
            tokens = word_tokenize(text_clean)
            tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
            pos_tags = pos_tag(tokens)
            lemmas = [
                self.lemmatizer.lemmatize(t, self._get_wordnet_pos(pos)) 
                for t, pos in pos_tags
            ]

            # n-grams
            ngram_tokens = lemmas.copy()
            if self.use_bigrams:
                ngram_tokens.extend(['_'.join(bg) for bg in ngrams(lemmas, 2)])
            if self.use_trigrams:
                ngram_tokens.extend(['_'.join(tg) for tg in ngrams(lemmas, 3)])
            
            all_tokens.append(ngram_tokens)
        
        return all_tokens

    def _avg_vector(self, tokens, model):
        """Calcula el vector promedio de Word2Vec"""
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

    def _load_glove(self):
        """Carga embeddings GloVe"""
        if not os.path.exists(self.glove_path):
            print(f"⚠️ ADVERTENCIA: No se encontró GloVe en {self.glove_path}")
            print("   Continuando sin embeddings GloVe...")
            self.glove = {}
            return
        
        print(f"📂 Cargando GloVe desde {self.glove_path}...")
        self.glove = {}
        
        with open(self.glove_path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.glove[word] = vector
        
        print(f"✅ GloVe cargado: {len(self.glove)} palabras")

    def _avg_glove(self, tokens):
        """Calcula el vector promedio de GloVe"""
        vecs = [self.glove[w] for w in tokens if w in self.glove]
        return np.mean(vecs, axis=0) if vecs else np.zeros(100)

    # ----------------------------
    # Fit y transform
    # ----------------------------
    def fit(self, X, y=None):
        """Entrena el preprocessor con los datos"""
        print("\n" + "="*60)
        print("🔄 Entrenando TextPreprocessor...")
        print("="*60)
        
        # Tokenizar
        print("\n📝 Tokenizando y generando n-gramas...")
        self.X_tokens_ = self._tokenize_series(X)

        # Word2Vec
        print("\n🤖 Entrenando Word2Vec...")
        self.w2v_model = Word2Vec(
            sentences=self.X_tokens_,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4  # Aumentado para mejor rendimiento
        )

        if self.w2v_model_path:
            os.makedirs(os.path.dirname(self.w2v_model_path), exist_ok=True)
            self.w2v_model.save(self.w2v_model_path)
            print(f"✅ Word2Vec guardado en {self.w2v_model_path}")

        # Cargar GloVe si existe
        if self.glove_path:
            self._load_glove()
        else:
            print("ℹ️  No se especificó ruta de GloVe, continuando sin él...")

        # Cargar SBERT
        print("\n🤖 Cargando modelo SBERT...")
        self.sbert_model = SentenceTransformer(self.sbert_model_name)
        
        print("\n" + "="*60)
        print("✅ TextPreprocessor entrenado exitosamente")
        print("="*60 + "\n")
        return self

    def transform(self, X):
        """Transforma textos en embeddings"""
        from sentence_transformers import SentenceTransformer
        
        # Cargar Word2Vec si no está en memoria
        if self.w2v_model is None:
            if self.w2v_model_path and os.path.exists(self.w2v_model_path):
                print(f"📂 Cargando Word2Vec desde {self.w2v_model_path}")
                self.w2v_model = Word2Vec.load(self.w2v_model_path)
            else:
                raise ValueError("❌ No se encontró el modelo Word2Vec. Ejecuta fit() primero.")

        if self.sbert_model is None:
            print(f"🤖 Cargando modelo SBERT: {self.sbert_model_name}")
            self.sbert_model = SentenceTransformer(self.sbert_model_name)

        # Tokenizar
        tokens = self._tokenize_series(X)
        
        # Generar embeddings
        X_w2v = np.array([self._avg_vector(t, self.w2v_model) for t in tokens])
        X_glove = np.array([self._avg_glove(t) for t in tokens]) if self.glove else None
        X_sbert = self.sbert_model.encode(X.tolist(), batch_size=32, show_progress_bar=False)

        return {'w2v': X_w2v, 'glove': X_glove, 'sbert': X_sbert}

    # ----------------------------
    # Serialización para joblib
    # ----------------------------
    def __getstate__(self):
        """Preparar el objeto para serialización con joblib"""
        state = self.__dict__.copy()
        
        # Remover objetos no serializables
        state['w2v_model'] = None
        state['sbert_model'] = None
        state['stop_words'] = None
        state['lemmatizer'] = None
        
        return state

    def __setstate__(self, state):
        """Restaurar el objeto después de deserialización con joblib"""
        # Restaurar estado
        self.__dict__.update(state)
        
        # Reinicializar modelos como None
        self.w2v_model = None
        self.sbert_model = None
        
        # Reinicializar recursos NLTK
        print("🔄 Reinicializando recursos NLTK después de deserialización...")
        self._ensure_nltk_resources()
        self._init_nltk_components()
        print("✅ TextPreprocessor deserializado correctamente")
