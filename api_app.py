# -*- coding: utf-8 -*-
"""
AIM - Intelligent Marketing API
Analyse automatique des données marketing d'une entreprise
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
    version="2.0.0",
    docs_url=None,   # on utilise custom docs
    redoc_url=None
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
        df = pd.read_csv(file.file, sep=None, engine="python", encoding="utf-8-sig")
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

# ----------------------------
# CUSTOM SWAGGER UI
# ----------------------------
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{app.title} - Swagger</title>
        <link type="text/css" rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.18.2/swagger-ui.css">
        <style>
            body {{ background-color: #FFF176 !important; }}
            .swagger-ui .topbar {{ background-color: #FFD54F !important; }}
            .swagger-ui .info {{ color: #333 !important; }}
            .swagger-ui .scheme-container {{ background-color: #FFF9C4 !important; }}
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.18.2/swagger-ui-bundle.js"></script>
        <script>
            const ui = SwaggerUIBundle({{
                url: '{app.openapi_url}',
                dom_id: '#swagger-ui',
                presets: [SwaggerUIBundle.presets.apis],
                layout: "BaseLayout"
            }})
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# ----------------------------
# CUSTOM REDOC
# ----------------------------
@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{app.title} - Redoc</title>
        <style>
            body {{ background-color: #FFF176 !important; }}
            .menu-content {{ background-color: #FFF9C4 !important; }}
        </style>
        <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
    </head>
    <body>
        <redoc spec-url="{app.openapi_url}"></redoc>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
