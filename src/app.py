import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from prometheus_client import start_http_server, Gauge, REGISTRY
import os

# --- 1. CONFIGURATION PROMETHEUS (MONITORING) ---
# @st.cache_resource empêche de relancer cette fonction à chaque clic
@st.cache_resource
def init_monitoring():
    # A. Démarrer le serveur HTTP (Port 8000)
    try:
        start_http_server(8000)
    except OSError:
        # Si le port est occupé, c'est que le serveur tourne déjà. On continue.
        pass

    # B. Le "Nettoyeur" (Pour éviter l'erreur Duplicated timeseries)
    metric_name = 'taxi_last_price_dollars'
    
    # Si la métrique existe déjà dans le registre global (mémoire Python), on la supprime
    if metric_name in REGISTRY._names_to_collectors:
        try:
            REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])
        except KeyError:
            pass # Sécurité supplémentaire

    # C. Création de la jauge toute neuve
    return Gauge(metric_name, 'Dernier prix estimé par le modèle')

# On initialise la jauge une bonne fois pour toutes
PRICE_GAUGE = init_monitoring()


# --- 2. CONFIGURATION SPARK ---
# On crée la session Spark (nécessaire pour créer le DataFrame d'entrée)
spark = SparkSession.builder \
    .appName("Taxi_App") \
    .master("local[*]") \
    .getOrCreate()


# --- 3. CHARGEMENT DU PIPELINE ---
@st.cache_resource
def load_pipeline():
    # On pointe vers le dossier du PIPELINE (pas juste le modèle)
    model_path = "/app/data/pipeline_final"
    
    if os.path.exists(model_path):
        # On utilise PipelineModel car on a sauvegardé un pipeline
        return PipelineModel.load(model_path)
    else:
        return None

model = load_pipeline()


# --- 4. INTERFACE STREAMLIT ---
st.title("🚖 NYC Taxi Price Predictor")
st.markdown("""
Cette application utilise un **Pipeline Spark** (VectorAssembler + LinearRegression) 
pour estimer le prix et envoie les données à **Grafana** en temps réel.
""")

# Formulaire de saisie
col1, col2 = st.columns(2)
with col1:
    distance = st.slider("Distance (miles)", 0.5, 100.0, 2.0, step=0.5)
with col2:
    hour = st.slider("Heure de la journée", 0, 23, 14)

day = st.selectbox("Jour de la semaine", [1, 2, 3, 4, 5, 6, 7], 
                   format_func=lambda x: ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"][x-1])

# Bouton de calcul
if st.button("Calculer le prix 🚀"):
    if model:
        try:
            # 1. Création des données brutes
            # Spark a besoin d'une liste de tuples
            input_data = spark.createDataFrame(
                [(float(distance), int(hour), int(day))],
                ["trip_distance", "hour", "day_of_week"]
            )
            
            # 2. Prédiction via le Pipeline
            # Le pipeline va automatiquement vectoriser les colonnes grâce au VectorAssembler intégré
            prediction = model.transform(input_data)
            
            # 3. Extraction du résultat
            price = prediction.select("prediction").first()[0]
            
            # 4. Affichage Streamlit
            st.success(f"💰 Prix estimé : **${price:.2f}**")
            
            # 5. Envoi à Prometheus/Grafana
            PRICE_GAUGE.set(price)
            st.caption(f"✅ La valeur {price:.2f} a été envoyée au monitoring.")
            
        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
    else:
        st.error("⚠️ Pipeline introuvable. As-tu bien lancé 'python src/train_model.py' ?")