# -*- coding: utf-8 -*-

"""
AIM - Intelligent Marketing API
Analyse automatique des données marketing d'une entreprise
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

# ----------------------------

# INITIALISATION API

# ----------------------------

app = FastAPI(
title="AIM - Intelligent Marketing API",
description="Analyse intelligente : spam, sentiment social, avis clients",
version="2.0.0"
)

# --- CORS pour Streamlit ---

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# ----------------------------

# LOGGING

# ----------------------------

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Logger initialisé avec succès")

# ----------------------------

# CHARGEMENT DES MODELES

# ----------------------------

try:
    youtube_model = joblib.load("model_youtube.sav")
    tweets_model = joblib.load("model_tweets.sav")
    reviews_model = joblib.load("model_reviews.sav")
    logger.info("Modèles chargés avec succès")
except Exception as e:
    logger.error(f"Erreur de chargement des modèles : {e}")

# ----------------------------

# SCHEMAS

# ----------------------------

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]

# ----------------------------

# DETECTION AUTOMATIQUE DES COLONNES TEXTUELLES

# ----------------------------

def detect_text_columns(df, min_avg_len: int = 20):
    """
    Retourne les colonnes textuelles pertinentes
    min_avg_len : longueur moyenne minimale pour considérer une colonne comme textuelle
    """
    cols = []
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len >= min_avg_len:
                cols.append(col)
                return cols

# ----------------------------

# CHOIX DU MODELE PAR NOM DE COLONNE

# ----------------------------

def guess_model(column_name):
    """
    Choisit automatiquement le modèle à utiliser selon le nom de la colonne
    """
    col = column_name.lower()
    
    if any(w in col for w in ["tweet", "post", "message", "social", "commentaire"]):
        return "tweets"
    if any(w in col for w in ["review", "avis", "feedback", "note", "commentaire_client"]):
        return "reviews"
    return "youtube"  # modèle par défaut (spam / faux avis)


# ----------------------------

# ENDPOINT GLOBAL D'ANALYSE

# ----------------------------

@app.post("/predict/auto")
async def predict_auto(file: UploadFile = File(...)):
    try:
        logger.info(f"Fichier reçu : {file.filename}")

        df = pd.read_csv(
        file.file,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )
        logger.info(f"Fichier chargé, {df.shape[0]} lignes, {df.shape[1]} colonnes")

        text_cols = detect_text_columns(df)
        logger.info(f"Colonnes textuelles détectées : {text_cols}")

        results = df.copy()

        for col in text_cols:
            model_name = guess_model(col)
            logger.info(f"Analyse de la colonne '{col}' avec le modèle '{model_name}'")

            if model_name == "youtube":
                    preds = youtube_model.predict(df[col].fillna("").astype(str))
                    results[col + "_spam"] = ["spam" if p == 1 else "ham" for p in preds]
            elif model_name == "tweets":
                preds = tweets_model.predict(df[col].fillna("").astype(str))
                results[col + "_tweet_sentiment"] = preds

            elif model_name == "reviews":
                preds = reviews_model.predict(df[col].fillna("").astype(str))
                results[col + "_review_sentiment"] = ["positif" if p == 1 else "negatif" for p in preds]

        return results.to_dict(orient="records")

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse automatique : {e}")
        return {"error": str(e)}

