"""
process/inverted.py — Construction de l'Index Inversé
======================================================
Construit la structure d'index inversé à partir d'une liste
de tokens normalisés.

Structure de l'index :
    {
        "mot": {
            "positions": [0, 5, 12],   ← positions dans le texte
            "freq": 3                   ← fréquence totale
        }
    }
"""

import json
import os
from collections import Counter


def build_inverted_index(tokens: list) -> dict:
    """
    Construit un index inversé depuis une liste de tokens.
    
    Args:
        tokens: liste de mots normalisés (dans l'ordre du texte)
    
    Returns:
        index inversé { mot: { positions: [...], freq: int } }
    
    Exemple :
        >>> build_inverted_index(['school', 'play', 'school'])
        {'school': {'positions': [0, 2], 'freq': 2},
         'play':   {'positions': [1],    'freq': 1}}
    """
    index = {}
    freq  = Counter(tokens)

    for position, word in enumerate(tokens):
        if word not in index:
            index[word] = {"positions": [], "freq": 0}
        index[word]["positions"].append(position)
        index[word]["freq"] = freq[word]

    # Trier par fréquence décroissante (mots les plus importants en premier)
    index = dict(sorted(index.items(), key=lambda x: x[1]["freq"], reverse=True))
    return index


def save_index(index: dict, path: str = "inverted_index.json") -> None:
    """Sauvegarde l'index inversé en JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"✅ Index sauvegardé : {path} ({len(index)} mots)")


def load_index(path: str = "inverted_index.json") -> dict:
    """Charge un index inversé depuis un fichier JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index non trouvé : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_index(index: dict, query: str) -> list:
    """
    Recherche dans l'index inversé.
    
    Args:
        index: index inversé
        query: requête textuelle (un ou plusieurs mots)
    
    Returns:
        liste de (position, score) triée par score décroissant
    """
    words  = query.lower().split()
    scores = {}

    for word in words:
        for key, data in index.items():
            # Correspondance exacte ou partielle
            if word in key or key in word:
                for pos in data["positions"]:
                    scores[pos] = scores.get(pos, 0) + data["freq"]

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    tokens = ["school", "play", "aaron", "school", "curious", "play", "life"]
    index  = build_inverted_index(tokens)

    print("Index inversé construit :")
    for word, data in index.items():
        print(f"  {word:<15} freq={data['freq']}  positions={data['positions']}")

    print("\nRecherche 'school' :")
    results = search_index(index, "school")
    print(f"  {results}")
