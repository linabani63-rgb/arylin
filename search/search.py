"""
search/search.py — Moteur de Recherche Unifié
=============================================
Interface unifiée pour la recherche textuelle et la recherche
par image (CBIR).

Classes :
    TextSearchEngine   : recherche dans l'index inversé
    ImageSearchEngine  : recherche par histogramme (CBIR)
    SearchEngine       : moteur unifié (texte + image)
"""

import json
import os
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  MOTEUR TEXTUEL
# ─────────────────────────────────────────────────────────────

class TextSearchEngine:
    """
    Moteur de recherche textuel basé sur l'index inversé.
    
    Usage :
        engine = TextSearchEngine("inverted_index.json")
        results = engine.search("school colombia")
    """

    def __init__(self, index_path: str = "inverted_index.json"):
        self.index = {}
        self.index_path = index_path
        self._load_index()

    def _load_index(self):
        """Charge l'index inversé depuis le fichier JSON."""
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
            print(f"✅ Index chargé : {len(self.index)} mots")
        else:
            print(f"⚠  Index non trouvé : {self.index_path}")

    def search(self, query: str, top_k: int = 20) -> list:
        """
        Recherche des frames correspondant à la requête.
        
        Algorithme :
            Pour chaque mot de la requête :
                Chercher dans l'index (correspondance exacte + partielle)
                Additionner les scores de fréquence
            Retourner les positions triées par score décroissant.
        
        Args:
            query: requête textuelle
            top_k: nombre max de résultats
        
        Returns:
            liste de { position, score, score_pct, frame, hist }
        """
        if not query.strip():
            return []

        words  = query.lower().split()
        scores = {}

        for word in words:
            for key, data in self.index.items():
                if word in key or key in word:
                    for pos in data["positions"]:
                        scores[pos] = scores.get(pos, 0) + data["freq"]

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_score = sorted_scores[0][1] if sorted_scores else 1

        results = []
        for pos, score in sorted_scores[:top_k]:
            results.append({
                "position":  pos,
                "score":     score,
                "score_pct": round((score / max_score) * 100),
                "frame":     f"frames/frame_{pos}.jpg",
                "hist":      f"histograms/hist_{pos}.png"
            })

        return results

    def get_stats(self) -> dict:
        """Retourne les statistiques de l'index."""
        return {
            "total_mots": len(self.index),
            "mots":       list(self.index.keys()),
            "top_10":     list(self.index.items())[:10]
        }


# ─────────────────────────────────────────────────────────────
#  MOTEUR IMAGE (CBIR)
# ─────────────────────────────────────────────────────────────

class ImageSearchEngine:
    """
    Moteur de recherche par contenu visuel (CBIR).
    Méthode : comparaison d'histogrammes RGB.
    
    Usage :
        engine = ImageSearchEngine("data/histograms_cache.json")
        results = engine.search("query.jpg")
    """

    def __init__(self, hist_cache_path: str = "data/histograms_cache.json"):
        self.hist_cache = {}
        self.cache_path = hist_cache_path
        self._load_cache()

    def _load_cache(self):
        """Charge le cache des histogrammes."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.hist_cache = json.load(f)
            print(f"✅ Histogrammes chargés : {len(self.hist_cache)} frames")
        else:
            print(f"⚠  Cache histogrammes non trouvé : {self.cache_path}")

    def _compute_histogram(self, image_path: str) -> list:
        """Calcule le vecteur histogramme RGB d'une image."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) requis pour la recherche par image")
        img = cv2.imread(image_path)
        if img is None:
            return None
        hist_vector = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_vector.extend(hist.tolist())
        return hist_vector

    def _distance(self, h1: list, h2: list) -> float:
        """
        Calcule la distance entre deux histogrammes.
        Méthode : corrélation inversée (0 = identiques, 1 = opposés).
        """
        v1 = np.array(h1, dtype=np.float32)
        v2 = np.array(h2, dtype=np.float32)
        corr = cv2.compareHist(v1, v2, cv2.HISTCMP_CORREL)
        return float(1.0 - ((corr + 1) / 2))

    def search(self, image_path: str, top_k: int = 20) -> list:
        """
        Cherche les frames les plus similaires à l'image requête.
        
        Args:
            image_path: chemin vers l'image requête
            top_k:      nombre max de résultats
        
        Returns:
            liste de { frame_id, distance, similarity_pct, frame, hist }
            triée par similarité décroissante
        """
        query_hist = self._compute_histogram(image_path)
        if query_hist is None:
            return []

        distances = []
        for frame_id, hist_data in self.hist_cache.items():
            dist = self._distance(query_hist, hist_data)
            distances.append((frame_id, dist))

        distances.sort(key=lambda x: x[1])

        results = []
        for frame_id, dist in distances[:top_k]:
            results.append({
                "frame_id":       int(frame_id),
                "distance":       round(dist, 4),
                "similarity_pct": round((1 - dist) * 100),
                "frame":          f"frames/frame_{frame_id}.jpg",
                "hist":           f"histograms/hist_{frame_id}.png"
            })

        return results


# ─────────────────────────────────────────────────────────────
#  MOTEUR UNIFIÉ
# ─────────────────────────────────────────────────────────────

class SearchEngine:
    """
    Moteur de recherche unifié : texte + image.
    
    Usage :
        engine = SearchEngine()
        results = engine.search_text("school")
        results = engine.search_image("query.jpg")
    """

    def __init__(self):
        self.text_engine  = TextSearchEngine()
        self.image_engine = ImageSearchEngine()

    def search_text(self, query: str, top_k: int = 20) -> list:
        return self.text_engine.search(query, top_k)

    def search_image(self, image_path: str, top_k: int = 20) -> list:
        return self.image_engine.search(image_path, top_k)


# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    engine = TextSearchEngine()

    query = input("Entrez un mot à chercher : ").strip()
    if query:
        results = engine.search(query)
        print(f"\n{len(results)} résultat(s) pour \"{query}\" :\n")
        for r in results[:5]:
            print(f"  Position {r['position']:3d}  |  Score {r['score']:3d}  "
                  f"({r['score_pct']}%)  →  {r['frame']}")
    else:
        stats = engine.get_stats()
        print(f"Index : {stats['total_mots']} mots")
        print("Mots indexés :", stats["mots"])
