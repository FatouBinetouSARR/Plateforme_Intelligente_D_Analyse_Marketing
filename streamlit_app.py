# ================================================================
# streamlit_app.py — Version avec présentation améliorée des opportunités
# AIM : Analyse Marketing Intelligente
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ================================================================
# 🎨 Palette couleurs AIM + Fond jaune clair
# ================================================================
AIM_PALETTE = [
    "#2ECC71", "#27AE60", "#3498DB", "#2980B9",
    "#F1C40F", "#F39C12", "#E67E22", "#E74C3C", "#C0392B"
]

# Configuration du fond jaune très clair
BACKGROUND_COLOR = "#FFFDE7"  # Jaune très clair et lumineux
SIDEBAR_COLOR = "#FFF9C4"    # Jaune un peu plus soutenu pour le sidebar
TEXT_COLOR = "#212121"       # Gris foncé pour meilleur contraste

# Appliquer le style CSS pour le fond clair et titre centré
page_bg_css = f"""
<style>
.stApp {{
    background-color: {BACKGROUND_COLOR} !important;
    color: {TEXT_COLOR} !important;
}}

/* TITRE PRINCIPAL CENTRÉ ET PLUS GROS */
h1 {{
    text-align: center !important;
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    color: #FF6B00 !important;
    margin-top: 20px !important;
    margin-bottom: 40px !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    background: linear-gradient(90deg, #FF6B00, #FF9800);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 15px;
    border-bottom: 4px solid #FFD54F;
}}

/* Sous-titres */
h2 {{
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #5D4037 !important;
    margin-top: 30px !important;
    margin-bottom: 20px !important;
    padding-bottom: 10px;
    border-bottom: 2px solid #FFD54F;
}}

h3 {{
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #795548 !important;
}}

/* Style pour les cartes d'opportunités */
.opportunity-card {{
    background: linear-gradient(135deg, #ffffff, #FFF9C4);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    border-left: 6px solid #FF9800;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.opportunity-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
}}

.opportunity-badge {{
    display: inline-block;
    background: linear-gradient(90deg, #FF9800, #FF5722);
    color: white;
    padding: 8px 15px;
    border-radius: 20px;
    font-weight: bold;
    margin-bottom: 10px;
    font-size: 0.9rem;
}}

.opportunity-tag {{
    display: inline-block;
    background: #E3F2FD;
    color: #1565C0;
    padding: 5px 12px;
    border-radius: 15px;
    margin: 3px;
    font-size: 0.85rem;
    border: 1px solid #90CAF9;
}}

/* Style pour le contenu principal */
.main .block-container {{
    background-color: rgba(255, 255, 255, 0.85) !important;
    border-radius: 15px;
    padding: 25px;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(255, 235, 59, 0.2);
}}

/* Style pour les cartes et sections */
.css-1d391kg, .css-12oz5g7, .css-1y4p8pa, .css-18e3th9, .css-1lcbmhc {{
    background-color: rgba(255, 255, 255, 0.92) !important;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(255, 235, 59, 0.3);
}}

/* Style pour les métriques */
.css-1xarl3l, .css-1v0mbdj, [data-testid="stMetric"] {{
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 10px;
    padding: 15px;
    border: 2px solid #FFEB3B !important;
    box-shadow: 0 3px 8px rgba(255, 193, 7, 0.15);
}}

/* Style pour le sidebar */
.css-1d391kg {{
    background-color: rgba(255, 253, 231, 0.95) !important;
}}
</style>
"""

# ================================================================
# ⚙️ Configuration Streamlit
# ================================================================
st.set_page_config(page_title="AIM – Dashboard", page_icon="📊", layout="wide")
st.markdown(page_bg_css, unsafe_allow_html=True)

# TITRE CENTRÉ AVEC MARKDOWN POUR UN MEILLUR CONTRÔLE
st.markdown("""
<div style="text-align: center;">
    <h1 style="font-size: 3.8rem; font-weight: 900; color: #FF6B00; 
               margin-bottom: 10px; text-shadow: 3px 3px 6px rgba(0,0,0,0.15);">
        📊 AIM – Analyse Marketing Intelligente
    </h1>
    <p style="font-size: 1.3rem; color: #666; margin-top: 0; margin-bottom: 40px;">
        Plateforme d'analyse avancée des sentiments et insights marketing
    </p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# 🔧 Fonctions utilitaires
# ================================================================
@st.cache_data(show_spinner=False)
def safe_load(filename):
    try:
        return joblib.load(filename)
    except:
        return None

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", " ", text)
    text = re.sub(r"[^a-z0-9àâäéèêëïîôöùûüç\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

# ================================================================
# 📥 Importation du fichier
# ================================================================

st.sidebar.header("1️⃣ Importer un Dataset")

uploaded = st.sidebar.file_uploader("Importer un CSV ou Excel", type=["csv", "xlsx"])

# Si aucun fichier n'est chargé → on arrête proprement
if uploaded is None:
    st.info("🗂️ Veuillez importer un fichier pour commencer.")
    st.stop()

# Lecture sécurisée du fichier
try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.success(f"✅ {uploaded.name} chargé avec succès !")

except Exception as e:
    st.error(f"❌ Erreur lors du chargement du fichier : {e}")
    st.stop()

# ================================================================
# 📌 Aperçu du dataset
# ================================================================
st.subheader("📌 Aperçu du dataset")
st.write(f"Nombre de lignes : **{df.shape[0]}**")
st.write(f"Nombre de colonnes : **{df.shape[1]}**")
st.dataframe(df.head(), use_container_width=True)

# ================================================================
# 🧹 Prétraitement automatique
# ================================================================
st.subheader("🧹 Prétraitement automatique du texte")
text_cols = df.select_dtypes(include=["object"]).columns.tolist()

if len(text_cols) == 0:
    st.error("Aucune colonne texte trouvée.")
    st.stop()

for col in text_cols:
    df[col] = df[col].astype(str).apply(clean_text)

df["clean_text"] = df[text_cols].agg(" ".join, axis=1)
st.success("Texte nettoyé ✔")

# ================================================================
# 🔠 Analyse des mots
# ================================================================
all_words = " ".join(df["clean_text"]).split()
words = [w for w in all_words if len(w) > 3 and w not in ENGLISH_STOP_WORDS]
wc = Counter(words)

freq_df = pd.DataFrame(wc.most_common(20), columns=["Mot", "Fréquence"])
fig_words = px.bar(freq_df, x="Mot", y="Fréquence", title="🔠 Top 20 des mots les plus fréquents", color_discrete_sequence=AIM_PALETTE)
st.plotly_chart(fig_words, use_container_width=True)

# ================================================================
# 🤖 Chargement modèles IA (optionnel)
# ================================================================
st.subheader("🤖 Chargement des modèles IA")
models = {
    "youtube": safe_load("model_youtube.sav"),
    "twitter": safe_load("model_tweets.sav"),
    "reviews": safe_load("model_reviews.sav")
}
vectorizers = {
    "youtube": safe_load("youtube_vectorizer.sav"),
    "twitter": safe_load("tweets_vectorizer.sav"),
    "reviews": safe_load("reviews_vectorizer.sav")
}

valid = [k for k in models if models[k] is not None and vectorizers[k] is not None]

# ================================================================
# 📡 Prédictions IA
# ================================================================
pred_cols = []
for k in valid:
    try:
        X = vectorizers[k].transform(df["clean_text"])
        df[f"pred_{k}"] = models[k].predict(X)
        pred_cols.append(f"pred_{k}")
    except:
        df[f"pred_{k}"] = np.nan

label_to_score = {"positive": 1, "neutral": 0, "negative": -1}

def fusion(row):
    scores = []
    for c in pred_cols:
        v = row[c]
        if pd.notnull(v): scores.append(label_to_score.get(str(v), 0))
    return np.mean(scores) if scores else 0

df["score_moyen"] = df.apply(fusion, axis=1)
df["sentiment"] = df["score_moyen"].apply(lambda s: "positive" if s>0 else "negative" if s<0 else "neutral")

# ================================================================
# 📊 KPIs
# ================================================================
st.header("📊 KPIs – Vue d'ensemble")

total = len(df)
pos = (df["sentiment"]=="positive").sum()
neut = (df["sentiment"]=="neutral").sum()
neg = (df["sentiment"]=="negative").sum()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total messages", total)
col2.metric("Positifs", pos)
col3.metric("Neutres", neut)
col4.metric("Négatifs", neg)
col5.metric("% Positif", f"{pos/total:.0%}")
col6.metric("Score AIM", f"{df['score_moyen'].mean():.2f}")

# ================================================================
# 📈 Graphiques
# ================================================================

# ------------------ 1️⃣ Top 20 des mots ------------------
st.subheader("🔠 Top 20 des mots les plus fréquents")

fig_words = px.bar(
    freq_df, x="Mot", y="Fréquence",
    title="🔠 Top 20 des mots les plus fréquents",
    color_discrete_sequence=AIM_PALETTE
)
st.plotly_chart(fig_words, use_container_width=True, key="fig_words_top20")

st.write("""
**Description :** Ce diagramme affiche les 20 mots les plus utilisés dans votre dataset après nettoyage.
Chaque barre représente le nombre d'occurrences d'un mot.
""")

top_word = freq_df.iloc[0]['Mot']
st.info(f"**Interprétation :** Le mot **'{top_word}'** est le plus fréquent. Cela peut indiquer le thème central des commentaires/messages.")

# ------------------ 2️⃣ Répartition des sentiments ------------------
st.subheader("📊 Répartition des sentiments")

fig_sent = px.pie(
    df, names="sentiment",
    title="Répartition des sentiments",
    color="sentiment",
    color_discrete_map={
        "positive": "#2ECC71",
        "neutral": "#F1C40F",
        "negative": "#E74C3C"
    }
)
st.plotly_chart(fig_sent, use_container_width=True, key="fig_sentiment")

st.write("""
**Description :** Ce diagramme circulaire montre la proportion de messages positifs, neutres et négatifs.
""")
st.info(f"**Interprétation :** {pos/total:.0%} des messages sont positifs, {neg/total:.0%} négatifs, et {neut/total:.0%} neutres. Cela permet d'évaluer rapidement la tonalité générale des messages.")

# ------------------ 3️⃣ Distribution du score de sentiment ------------------
st.subheader("📈 Distribution du score de sentiment")

fig_score = px.histogram(
    df, x="score_moyen", nbins=30,
    title="Distribution du score de sentiment",
    color_discrete_sequence=AIM_PALETTE
)
st.plotly_chart(fig_score, use_container_width=True, key="fig_score_distribution")

st.write("""
**Description :** L'histogramme montre la distribution des scores moyens de sentiment, allant de -1 (très négatif) à +1 (très positif).
""")
mean_score = df['score_moyen'].mean()
st.info(f"**Interprétation :** La moyenne des scores est {mean_score:.2f}, ce qui indique une tendance globale {'positive' if mean_score>0 else 'négative' if mean_score<0 else 'neutre'}.")

# ------------------ 4️⃣ Heatmap : influence des mots ------------------
st.subheader("🔥 Influence des mots-clés sur le sentiment")

heat_df = pd.DataFrame({
    w: [df[df["clean_text"].str.contains(w)]["score_moyen"].mean()]
    for w, _ in wc.most_common(20)
})

fig_heat = px.imshow(
    heat_df,
    labels=dict(x="Mot", y="", color="Score moyen"),
    x=heat_df.columns,
    y=["Score moyen"],
    color_continuous_scale="RdYlGn",
    title="🔥 Influence des mots-clés sur le sentiment"
)
st.plotly_chart(fig_heat, use_container_width=True, key="fig_heatmap")

st.write("""
**Description :** La heatmap montre l'influence moyenne de chaque mot-clé sur le sentiment global.
Les couleurs vertes indiquent un impact positif, les rouges un impact négatif.
""")
max_word = heat_df.idxmax(axis=1)[0]
min_word = heat_df.idxmin(axis=1)[0]
st.info(f"**Interprétation :** Le mot **'{max_word}'** est associé au sentiment le plus positif, tandis que **'{min_word}'** est le plus négatif. Cela permet d'identifier les points forts et faibles dans les messages.")

# ================================================================
# 🎯 Recommandations Marketing
# ================================================================
st.header("🎯 Recommandations Marketing AIM")

if pos/total > 0.50:
    st.success("✔ Excellent taux positif. Optimisez vos campaines existantes et amplifiez les points forts.")

if neg/total > 0.30:
    st.error("⚠ Beaucoup de commentaires négatifs → Action immédiate requise.")
    st.write("- Analysez les sources de frustration.\n- Améliorez l'expérience utilisateur.\n- Augmentez votre support client.")

if neut/total > 0.40:
    st.info("ℹ Forte neutralité : améliorez l'engagement et la clarté du message.")

# ================================================================
# 🎪 OPPORTUNITÉS MARKETING - NOUVELLE PRÉSENTATION
# ================================================================
st.write("---")
st.header("🎪 Opportunités Marketing Détectées")

# Création de colonnes pour une présentation en grille
top_words = wc.most_common(15)

# Déterminer la catégorie d'opportunité basée sur la fréquence et le sentiment
def get_opportunity_type(word, freq, total_words):
    freq_percentage = (freq / total_words) * 100
    
    if freq_percentage > 5:
        return "🔥 Hot Trend", "#FF5722"
    elif freq_percentage > 2:
        return "📈 Opportunity", "#FF9800"
    elif freq_percentage > 1:
        return "💡 Emerging", "#4CAF50"
    else:
        return "🔍 Niche", "#2196F3"

total_words_count = sum(wc.values())

# Diviser en 3 colonnes
cols = st.columns(3)

for idx, (mot, freq) in enumerate(top_words):
    freq_percentage = (freq / total_words_count) * 100
    opp_type, color = get_opportunity_type(mot, freq, total_words_count)
    
    with cols[idx % 3]:
        # Déterminer l'icône basée sur le type
        icon = "🔥" if "Hot" in opp_type else "📈" if "Opportunity" in opp_type else "💡" if "Emerging" in opp_type else "🔍"
        
        # Créer une carte d'opportunité
        st.markdown(f"""
        <div class="opportunity-card">
            <div class="opportunity-badge" style="background: {color}">
                {icon} {opp_type}
            </div>
            <h3 style="color: #333; margin: 10px 0;">{mot.capitalize()}</h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 15px 0;">
                <span style="font-size: 2rem; font-weight: bold; color: {color}">{freq}</span>
                <span style="background: #E8F5E8; color: #2E7D32; padding: 5px 10px; border-radius: 10px; font-weight: bold;">
                    {freq_percentage:.1f}%
                </span>
            </div>
            <div style="margin-top: 10px;">
                <span class="opportunity-tag">#{mot}</span>
                <span class="opportunity-tag">Marketing</span>
                <span class="opportunity-tag">Engagement</span>
            </div>
            <div style="margin-top: 15px; font-size: 0.9rem; color: #666;">
                <strong>💡 Action recommandée :</strong> Intégrer ce mot-clé dans vos campagnes de contenu.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Graphique supplémentaire pour les opportunités
st.subheader("📊 Distribution des Opportunités par Fréquence")

opp_df = pd.DataFrame(top_words, columns=["Mot", "Fréquence"])
opp_df["Pourcentage"] = (opp_df["Fréquence"] / total_words_count * 100).round(1)

fig_opp = px.treemap(
    opp_df, 
    path=["Mot"], 
    values="Fréquence",
    title="📊 Carte des Opportunités Marketing",
    color="Pourcentage",
    color_continuous_scale="YlOrRd",
    hover_data=["Pourcentage"]
)
st.plotly_chart(fig_opp, use_container_width=True)

# Recommandations synthétiques
st.info("""
**📋 Synthèse des Opportunités :**
- **Mots en forte croissance** : Points d'entrée pour de nouvelles campagnes
- **Mots émergents** : Surveiller pour anticiper les tendances  
- **Mots niche** : Opportunités de spécialisation et différenciation
""")

# ================================================================
# 💾 Export
# ================================================================
st.sidebar.header("💾 Exporter les résultats")
st.sidebar.download_button("Télécharger les résultats (CSV)", df.to_csv(index=False).encode(), "AIM_results.csv")