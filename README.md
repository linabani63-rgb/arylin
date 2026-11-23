# ARYLIN — Moteur de Recherche Multimédia (Version Dynamique)

> Uploadez n'importe quelle vidéo → tout s'indexe automatiquement → recherchez !

## Installation & Lancement

```bash
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:5000
```

## Utilisation

1. **Uploadez** votre vidéo sur la page d'accueil (glisser-déposer ou clic)
2. **Attendez** l'indexation automatique (pipeline avec barre de progression)
3. **Recherchez** par mot-clé ou par image

## Pipeline Automatique

```
Vidéo uploadée
    ├── Extraction frames (1 frame/s)
    ├── Détection keyframes (diff inter-frames)
    ├── Extraction audio (WAV)
    ├── Transcription (Google Speech API)
    ├── Histogrammes RGB (CBIR)
    └── Index inversé (NLP : tokenisation → stopwords → lemmatisation)
```

## Recherches

| Mode | Route | Description |
|---|---|---|
| Textuelle | `GET /search/text?q=mot` | Recherche dans l'index inversé (transcription) |
| Par image | `POST /search/image` | CBIR : comparaison d'histogrammes RGB |
| Info index | `GET /api/info` | Statistiques de l'index courant |
| Statut | `GET /pipeline/status` | Progression du pipeline en temps réel |

## Structure

```
arylin_v2/
├── app.py                  ← Application Flask (pipeline + API + routes)
├── process/
│   ├── normalisation.py    ← Tokenisation, stopwords, lemmatisation
│   ├── inverted.py         ← Construction index inversé
│   ├── ponderation.py      ← TF, TF-IDF
│   └── etalage.py          ← Formatage résultats
├── search/
│   └── search.py           ← Moteurs de recherche (texte + CBIR)
├── templates/index.html    ← Interface web (upload + pipeline + recherche)
└── requirements.txt
```

## Dépendances

Flask · OpenCV · MoviePy · SpeechRecognition · NumPy · Matplotlib · NLTK
