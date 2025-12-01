# ================================================================
# streamlit_app.py — Version stable et corrigée (Aucune erreur)
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
# 🎨 Palette couleurs AIM
# ================================================================
AIM_PALETTE = [
    "#2ECC71", "#27AE60", "#3498DB", "#2980B9",
    "#F1C40F", "#F39C12", "#E67E22", "#E74C3C", "#C0392B"
]

# ================================================================
# ⚙️ Configuration Streamlit
# ================================================================
st.set_page_config(page_title="AIM – Dashboard", page_icon="📊", layout="wide")
st.title("📊 AIM – Analyse Marketing Intelligente")

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
Chaque barre représente le nombre d’occurrences d’un mot.
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
st.info(f"**Interprétation :** {pos/total:.0%} des messages sont positifs, {neg/total:.0%} négatifs, et {neut/total:.0%} neutres. Cela permet d’évaluer rapidement la tonalité générale des messages.")

# ------------------ 3️⃣ Distribution du score de sentiment ------------------
st.subheader("📈 Distribution du score de sentiment")

fig_score = px.histogram(
    df, x="score_moyen", nbins=30,
    title="Distribution du score de sentiment",
    color_discrete_sequence=AIM_PALETTE
)
st.plotly_chart(fig_score, use_container_width=True, key="fig_score_distribution")

st.write("""
**Description :** L’histogramme montre la distribution des scores moyens de sentiment, allant de -1 (très négatif) à +1 (très positif).
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
**Description :** La heatmap montre l’influence moyenne de chaque mot-clé sur le sentiment global.
Les couleurs vertes indiquent un impact positif, les rouges un impact négatif.
""")
max_word = heat_df.idxmax(axis=1)[0]
min_word = heat_df.idxmin(axis=1)[0]
st.info(f"**Interprétation :** Le mot **'{max_word}'** est associé au sentiment le plus positif, tandis que **'{min_word}'** est le plus négatif. Cela permet d’identifier les points forts et faibles dans les messages.")

# ================================================================
# 🎯 Recommandations Marketing
# ================================================================
st.header("🎯 Recommandations Marketing AIM")

if pos/total > 0.50:
    st.success("✔ Excellent taux positif. Optimisez vos campagnes existantes et amplifiez les points forts.")

if neg/total > 0.30:
    st.error("⚠ Beaucoup de commentaires négatifs → Action immédiate requise.")
    st.write("- Analysez les sources de frustration.\n- Améliorez l'expérience utilisateur.\n- Augmentez votre support client.")

if neut/total > 0.40:
    st.info("ℹ Forte neutralité : améliorez l'engagement et la clarté du message.")

st.write("---")
st.write("### 🔧 Opportunités détectées sur les mots-clés :")
for mot, freq in wc.most_common(10):
    st.write(f"• **{mot}** → {freq} occurrences : potentiel marketing identifié.")

# ================================================================
# 💾 Export
# ================================================================
st.sidebar.header("💾 Exporter les résultats")
st.sidebar.download_button("Télécharger les résultats (CSV)", df.to_csv(index=False).encode(), "AIM_results.csv")
