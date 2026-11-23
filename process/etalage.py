"""
process/etalage.py — Affichage / Formatage des Résultats
=========================================================
Fonctions utilitaires pour formater et afficher les résultats
de recherche dans la console ou pour l'API.
"""

import json


def format_results_text(results: list, query: str, transcription: str = "") -> str:
    """
    Formate les résultats de recherche textuelle pour affichage console.
    
    Args:
        results:       liste de { position, score }
        query:         requête initiale
        transcription: texte complet (pour extraire le contexte)
    
    Returns:
        chaîne formatée pour la console
    """
    lines = []
    lines.append("=" * 55)
    lines.append(f"  Résultats pour : \"{query}\"")
    lines.append(f"  {len(results)} résultat(s) trouvé(s)")
    lines.append("=" * 55)

    words = transcription.split() if transcription else []

    for i, (pos, score) in enumerate(results[:10]):
        # Extraire le contexte (mots autour de la position)
        start   = max(0, pos - 4)
        end     = min(len(words), pos + 5)
        context = " ".join(words[start:end]) if words else f"Position {pos}"

        lines.append(f"\n  #{i+1}  Position {pos}  |  Score : {score}")
        lines.append(f"       Frame  : frames/frame_{pos}.jpg")
        lines.append(f"       Histo  : static/histograms/hist_{pos}.png")
        lines.append(f"       Extrait: ...{context}...")

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


def format_results_image(results: list) -> str:
    """
    Formate les résultats de recherche par image pour affichage console.
    
    Args:
        results: liste de { frame_id, distance, similarity_pct }
    
    Returns:
        chaîne formatée
    """
    lines = []
    lines.append("=" * 55)
    lines.append("  Résultats — Recherche par Image (CBIR)")
    lines.append(f"  Méthode : Histogramme RGB | {len(results)} résultat(s)")
    lines.append("=" * 55)

    for i, r in enumerate(results[:10]):
        bar_len = int(r.get("similarity_pct", 0) / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(
            f"\n  #{i+1}  Frame {r['frame_id']:4d}  "
            f"Similarité : {r.get('similarity_pct', 0):3d}%  [{bar}]"
        )
        lines.append(f"       Distance   : {r.get('distance', 0):.4f}")
        lines.append(f"       Fichier    : frames/frame_{r['frame_id']}.jpg")

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


def print_index_stats(inverted_index: dict) -> None:
    """Affiche les statistiques de l'index inversé dans la console."""
    total_positions = sum(d["freq"] for d in inverted_index.values())

    print("=" * 55)
    print("  Statistiques de l'Index Inversé")
    print("=" * 55)
    print(f"  Mots uniques    : {len(inverted_index)}")
    print(f"  Positions total : {total_positions}")
    print()
    print("  Top 10 termes :")
    for i, (word, data) in enumerate(list(inverted_index.items())[:10]):
        bar = "█" * min(data["freq"], 20)
        print(f"  {i+1:2d}. {word:<20} {data['freq']:3d}  {bar}")
    print("=" * 55)


# ── Test rapide ────────────────────────────────────────────
if __name__ == "__main__":
    # Simuler des résultats
    sample_results = [(5, 8), (12, 5), (0, 3), (26, 3)]
    sample_text    = ("I like to play Aaron and say that im a kindergarten "
                      "Dropbox but actually elementary school in Colombia")

    print(format_results_text(sample_results, "school colombia", sample_text))
