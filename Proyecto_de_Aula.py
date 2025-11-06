"""
Streamlit app: Análisis de sentimientos para 20 opiniones
Requisitos implementados:
- Subir archivo CSV con al menos 20 opiniones (columna 'text' o seleccionable)
- Preprocesamiento: limpieza, eliminación stopwords, lematización (usa spaCy si está instalado)
- Wordcloud de las opiniones
- Gráfico de barras con las 10 palabras más frecuentes
- Gráfico adicional: distribución de probabilidades de sentimiento (scores)
- Clasificador de sentimiento usando HuggingFace pipeline (por defecto intenta 'tabularisai/multilingual-sentiment-analysis')
- Tabla con resultado de las 20 opiniones y gráfico de porcentaje por clase
- Campo para enviar comentarios nuevos y recibir clasificación + respuesta sugerida

Notas de instalación (ejecutar en entorno):
pip install streamlit pandas scikit-learn matplotlib plotly wordcloud transformers torch spacy nltk
# Si trabajas en español con spaCy, instala modelo: python -m spacy download es_core_news_sm

Guardar este archivo como streamlit_sentiment_app.py y ejecutar:
streamlit run streamlit_sentiment_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import re

# Intentar spaCy para lematización; si no está disponible, usar fallback simple
try:
    import spacy
    nlp = None
    try:
        nlp = spacy.load("es_core_news_sm")
    except Exception:
        try:
            # intenta cargar inglés si español no existe
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
except Exception:
    spacy = None
    nlp = None

# Descargar stopwords NLTK si es necesario
nltk_downloaded = False
try:
    stopwords.words('spanish')
    nltk_downloaded = True
except Exception:
    nltk.download('stopwords')
    nltk_downloaded = True

# Preparar stopwords (español + inglés)
STOPWORDS = set()
try:
    STOPWORDS.update(stopwords.words('spanish'))
except Exception:
    pass
try:
    STOPWORDS.update(stopwords.words('english'))
except Exception:
    pass

# Funciones de preprocesamiento
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    if nlp is not None:
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if token.lemma_ not in STOPWORDS and not token.is_punct and not token.is_space]
        return " ".join(lemmas)
    else:
        # Fallback simple: devolver palabras sin stopwords
        tokens = [t for t in text.split() if t not in STOPWORDS]
        return " ".join(tokens)

@st.cache_resource
def get_hf_pipeline(model_name="tabularisai/multilingual-sentiment-analysis"):
    try:
        pipe = pipeline("text-classification", model=model_name, return_all_scores=True)
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo {model_name}: {e}")
        pipe = None
    return pipe

# App
st.set_page_config(page_title="Análisis de Sentimientos - 20 Opiniones", layout="wide")
st.title("Análisis de Opiniones Domicilios de Restaurante")
st.markdown("Sube un archivo .csv con al menos 20 opiniones")

col1, col2 = st.columns([2,1])
with col1:
    uploaded_file = st.file_uploader("Selecciona archivo CSV", type=["csv"]) 
    
with col2:
    model_choice = "tabularisai/multilingual-sentiment-analysis"
    st.markdown("\n")

df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        # intentar con otro encoding
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1')

if df is not None:
    st.subheader("Datos cargados")
    st.write("Vista previa de las primeras filas:")
    st.dataframe(df.head(10))

# Seleccionar columna de texto
    text_col = None
    possible_text_cols = [c for c in df.columns if df[c].dtype == object]
    if len(possible_text_cols) == 0:
        st.error("No se encontraron columnas de texto. Asegúrate de que el archivo CSV tenga una columna con opiniones en texto.")
    else:
        text_col = st.selectbox("Selecciona la columna de texto", options=possible_text_cols, index=0)


    # Tomar sólo primeras 20 opiniones
    
    if text_col:
        df['text_raw'] = df[text_col].astype(str)
        df = df.reset_index(drop=True)
        if df.shape[0] < 20:
            st.warning("El dataset tiene menos de 20 opiniones. Se trabajará con las disponibles.")
        df_20 = df['text_raw'].iloc[:20].copy()

        st.markdown("---")
        st.subheader("Procesamiento")
        # limpieza y lematización
        df_proc = pd.DataFrame()
        df_proc['original'] = df_20
        df_proc['clean'] = df_proc['original'].apply(clean_text)
        df_proc['lemmatized'] = df_proc['clean'].apply(lemmatize_text)

        st.write(df_proc[['original','lemmatized']])

        # Generar nube de palabras
        all_text = " ".join(df_proc['lemmatized'].astype(str).tolist())
        if all_text.strip() == "":
            st.warning("Después del preprocesamiento no quedaron palabras para generar la nube. Revisa stopwords/lematización.")
        else:
            wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, collocations=False).generate(all_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10,5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.subheader("Nube de palabras")
            st.pyplot(fig_wc)

            # Top 10 palabras
            tokens = [t for t in all_text.split() if t not in STOPWORDS and len(t)>1]
            counts = Counter(tokens)
            most10 = counts.most_common(10)
            words, freqs = zip(*most10) if most10 else ([],[])
            fig_bar = px.bar(x=list(words), y=list(freqs), labels={'x':'Palabra','y':'Frecuencia'}, title='Top 10 palabras')
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("Clasificación de sentimientos")
        st.write("Selecciona el modelo HF y presiona 'Clasificar' para procesar las 20 opiniones.")
        if st.button("Clasificar opiniones (20)"):
            with st.spinner("Cargando modelo y clasificando..."):
                pipe = get_hf_pipeline(model_choice)
                if pipe is None:
                    st.error("No se pudo cargar el pipeline de HF. Verifica conexión e instalación de dependencias.")
                else:
                    results = []
                    # pipe devuelve lista de scores por etiqueta (return_all_scores=True)
                    for text in df_proc['original']:
                        try:
                            out = pipe(text)
                            # out es lista de dicts con label y score
                            # convertimos a dict label->score
                            if isinstance(out, list) and len(out)>0 and isinstance(out[0], list):
                                scores_list = out[0]
                            else:
                                scores_list = out
                            label = max(scores_list, key=lambda x: x['score'])['label']
                            # crear dict de scores
                            score_dict = {d['label']: d['score'] for d in scores_list}
                        except Exception as e:
                            label = 'ERROR'
                            score_dict = {}
                        results.append((label, score_dict))

                    df_proc['label'] = [r[0] for r in results]
                    df_proc['scores'] = [r[1] for r in results]

                    # Mostrar tabla
                    st.write(df_proc[['original','label']])

                    # Gráfico porcentaje por clase
                    counts = df_proc['label'].value_counts().reset_index()
                    counts.columns = ['label','count']
                    counts['percent'] = 100 * counts['count'] / counts['count'].sum()
                    fig_pie = px.pie(counts, names='label', values='count', title='Porcentaje de opiniones por clase')
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Gráfico adicional: distribución de scores para la etiqueta ganadora (si existe probabilidad positiva)
                    # Extraer la probabilidad de la etiqueta escogida
                    probs = []
                    for s in df_proc['scores']:
                        if isinstance(s, dict):
                            probs.append(max(s.values()))
                        elif isinstance(s, list):
                            probs.append(max([d['score'] for d in s]))
                        else:
                            probs.append(np.nan)
                    df_proc['max_prob'] = probs
                    fig_hist = px.histogram(df_proc, x='max_prob', nbins=10, title='Distribución de probabilidades (score máximo por opinión)')
                    st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.subheader("Enviar un comentario nuevo")
        new_text = st.text_area("Escribe un comentario nuevo para clasificarlo", height=120)
        if st.button("Clasificar comentario" ):
            if not new_text or new_text.strip()=="":
                st.warning("Escribe primero un comentario para clasificar.")
            else:
                pipe = get_hf_pipeline(model_choice)
                if pipe is None:
                    st.error("No se pudo cargar el pipeline de HF.")
                else:
                    with st.spinner("Clasificando..."):
                        try:
                            out = pipe(new_text)
                            # manejar formato
                            if isinstance(out, list) and len(out)>0 and isinstance(out[0], list):
                                scores_list = out[0]
                            else:
                                scores_list = out
                            chosen = max(scores_list, key=lambda x: x['score'])
                            label = chosen['label']
                            score = chosen['score']
                        except Exception as e:
                            st.error(f"Error al clasificar: {e}")
                            label = None
                            score = None

                    if label is not None:
                        st.success(f"Sentimiento: {label}  (score={score:.2f})")
                        # Mensajes sugeridos según etiqueta (personalizables)
                        suggestions = {
                            'POSITIVE': 'Respuesta sugerida: ¡Gracias por tu comentario! Nos alegra que hayas tenido una buena experiencia. ¿Hay algo más en lo que podamos ayudar?',
                            'NEGATIVE': 'Respuesta sugerida: Lamentamos que hayas tenido una mala experiencia. Por favor contáctanos con detalles para resolverlo lo antes posible.',
                            'NEUTRAL': 'Respuesta sugerida: Gracias por tu comentario. Si deseas, cuéntanos más para mejorar.'
                        }
                        # intenta mapear algunas etiquetas multilingües
                        if label.upper() in suggestions:
                            st.info(suggestions[label.upper()])
                        elif 'NEG' in label.upper():
                            st.info(suggestions['NEGATIVE'])
                        elif 'POS' in label.upper():
                            st.info(suggestions['POSITIVE'])
                        else:
                            st.info(suggestions.get('NEUTRAL'))

     




        


#https://appopiniones-mqcekk8sjpymnxx39splef.streamlit.app/