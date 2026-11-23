"""
process/normalisation.py — Normalisation du texte
==================================================
Module de prétraitement linguistique.

Fonctions exportées :
    normalize_text(text)   → liste de tokens normalisés
    tokenize(text)         → liste de tokens bruts
    remove_stopwords(tokens) → tokens filtrés
"""

import re
from collections import Counter

# ── Stopwords ──────────────────────────────────────────────
STOPWORDS = {
    "the", "is", "and", "a", "to", "of", "in", "on", "for", "an",
    "this", "that", "it", "its", "as", "be", "was", "were", "are",
    "but", "or", "not", "with", "at", "by", "from", "have", "had",
    "has", "do", "did", "does", "so", "up", "out", "if", "then",
    "i", "me", "my", "we", "you", "he", "she", "they", "them",
    "just", "all", "about", "into", "more", "also", "been", "will",
    "would", "could", "should", "very", "than", "when", "which"
}


def tokenize(text: str) -> list:
    """
    Tokenise un texte brut.
    Étapes : minuscules → suppression ponctuation → découpage.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def remove_stopwords(tokens: list, extra_stopwords: set = None) -> list:
    """
    Supprime les mots vides (stopwords) d'une liste de tokens.
    
    Args:
        tokens: liste de mots
        extra_stopwords: stopwords supplémentaires (optionnel)
    
    Returns:
        liste filtrée
    """
    stop = STOPWORDS.copy()
    if extra_stopwords:
        stop.update(extra_stopwords)
    return [w for w in tokens if w not in stop and len(w) > 1]


def lemmatize(tokens: list) -> list:
    """
    Lemmatisation des tokens (réduction à la forme canonique).
    Utilise NLTK si disponible, sinon retourne les tokens tels quels.
    """
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(w) for w in tokens]
    except ImportError:
        return tokens


def normalize_text(text: str) -> list:
    """
    Pipeline complet de normalisation.
    
    Args:
        text: texte brut (transcription)
    
    Returns:
        liste de tokens normalisés (tokenisés + filtrés + lemmatisés)
    
    Exemple :
        >>> normalize_text("I was playing with Aaron at school")
        ['play', 'aaron', 'school']
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens


def compute_tf(tokens: list) -> dict:
    """
    Calcule la fréquence de chaque terme (Term Frequency).
    
    Returns:
        dict { mot: fréquence }
    """
    return dict(Counter(tokens))


# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    sample = "I like to play and say that I was at school in Colombia"
    result = normalize_text(sample)
    print("Texte original :", sample)
    print("Normalisé      :", result)
    print("TF             :", compute_tf(result))
