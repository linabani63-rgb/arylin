"""
process/ponderation.py — Pondération TF-IDF
============================================
Calcule le poids de chaque terme selon la méthode TF-IDF.

    TF  (Term Frequency)  = fréquence du terme dans le document
    IDF (Inverse Document Frequency) = rareté du terme dans la collection

    TF-IDF = TF × IDF  (plus la valeur est haute, plus le terme est important)

Note : pour un seul document (une seule vidéo), on utilise TF uniquement.
       IDF devient pertinent avec plusieurs vidéos.
"""

import math
from collections import Counter


def compute_tf(tokens: list) -> dict:
    """
    Calcule la fréquence de terme (TF) brute.
    
    Returns:
        { mot: fréquence_absolue }
    """
    return dict(Counter(tokens))


def compute_tf_normalized(tokens: list) -> dict:
    """
    Calcule la fréquence de terme (TF) normalisée (0 à 1).
    TF_norm = freq / total_tokens
    
    Returns:
        { mot: fréquence_normalisée }
    """
    freq = Counter(tokens)
    total = len(tokens)
    return {w: count / total for w, count in freq.items()}


def compute_idf(documents: list) -> dict:
    """
    Calcule l'IDF sur une collection de documents.
    IDF = log(N / df)  avec N = nb docs, df = nb docs contenant le terme.
    
    Args:
        documents: liste de listes de tokens
    
    Returns:
        { mot: idf_value }
    """
    N = len(documents)
    df = {}  # document frequency

    for doc in documents:
        for word in set(doc):
            df[word] = df.get(word, 0) + 1

    idf = {}
    for word, count in df.items():
        idf[word] = math.log(N / count) if count > 0 else 0

    return idf


def compute_tfidf(tokens: list, documents: list = None) -> dict:
    """
    Calcule TF-IDF pour un document.
    Si documents=None, retourne uniquement TF.
    
    Args:
        tokens:    tokens du document courant
        documents: collection complète (pour IDF)
    
    Returns:
        { mot: tfidf_score }
    """
    tf = compute_tf_normalized(tokens)

    if documents is None or len(documents) <= 1:
        # Un seul document → TF uniquement
        return tf

    idf = compute_idf(documents)
    return {w: tf_val * idf.get(w, 1.0) for w, tf_val in tf.items()}


# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    tokens = ["school", "play", "aaron", "school", "curious", "play",
              "school", "life", "colombia", "question"]

    tf     = compute_tf(tokens)
    tf_n   = compute_tf_normalized(tokens)

    print("TF brute :")
    for w, v in sorted(tf.items(), key=lambda x: -x[1]):
        print(f"  {w:<15} {v}")

    print("\nTF normalisée :")
    for w, v in sorted(tf_n.items(), key=lambda x: -x[1]):
        print(f"  {w:<15} {v:.4f}")
