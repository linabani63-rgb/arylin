"""
ARYLIN — Moteur de Recherche Multimédia (Version Dynamique)
============================================================
Upload n'importe quelle vidéo → extraction automatique →
indexation → recherche textuelle + recherche par image.
"""

import os, json, re, threading, time, uuid, shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from flask import (Flask, render_template, request,
                   jsonify, send_from_directory)

from object_detection import (
    detect_objects, build_object_index,
    search_objects, SEARCH_KEYWORDS
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# ─────────────────────────────────────────────
#  CHEMINS
# ─────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR    = os.path.join(BASE, 'videos')
FRAMES_DIR    = os.path.join(BASE, 'frames')
KEYFRAMES_DIR = os.path.join(BASE, 'keyframes')
AUDIOS_DIR    = os.path.join(BASE, 'audios')
TEXTE_DIR     = os.path.join(BASE, 'texte')
DATA_DIR      = os.path.join(BASE, 'data')
HIST_STATIC   = os.path.join(BASE, 'static', 'histograms')
FRAME_STATIC  = os.path.join(BASE, 'static', 'frames')

for d in [VIDEOS_DIR, FRAMES_DIR, KEYFRAMES_DIR, AUDIOS_DIR,
          TEXTE_DIR, DATA_DIR, HIST_STATIC, FRAME_STATIC]:
    os.makedirs(d, exist_ok=True)

INDEX_FILE    = os.path.join(BASE, 'inverted_index.json')
HIST_CACHE    = os.path.join(DATA_DIR, 'histograms_cache.json')
TRANSCRIPT    = os.path.join(TEXTE_DIR, 'transcription.txt')
OBJ_INDEX     = os.path.join(DATA_DIR, 'object_index.json')

# ─────────────────────────────────────────────
#  ÉTAT GLOBAL DU PIPELINE
# ─────────────────────────────────────────────
pipeline_status = {
    "running": False,
    "step": "",
    "progress": 0,
    "total": 0,
    "done": False,
    "error": None,
    "video_name": ""
}

# ─────────────────────────────────────────────
#  STOPWORDS
# ─────────────────────────────────────────────
STOPWORDS = {
    "the","is","and","a","to","of","in","on","for","an","this","that",
    "it","its","as","be","was","were","are","but","or","not","with",
    "at","by","from","have","had","has","do","did","does","so","up",
    "out","if","then","i","me","my","we","you","he","she","they","them",
    "just","all","about","into","more","also","been","will","would",
    "could","should","very","than","when","which","im","its","dont",
    "cant","wont","thats","its","ive","id","ill"
}

# ─────────────────────────────────────────────────────────────
#  PIPELINE D'INDEXATION
# ─────────────────────────────────────────────────────────────

def set_status(step, progress=0, total=0):
    pipeline_status["step"]     = step
    pipeline_status["progress"] = progress
    pipeline_status["total"]    = total

def normalize_text(text):
    """Tokenisation + stopwords + lemmatisation simple."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    filtered = [w for w in tokens if w not in STOPWORDS and len(w) > 1]

    # Lemmatisation avec NLTK si disponible
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        filtered = [lem.lemmatize(w) for w in filtered]
    except Exception:
        pass

    return filtered

def extract_frames_from_video(video_path, fps_target=1):
    """Extrait les frames à fps_target frames/seconde."""
    # Nettoyer frames précédentes
    for f in os.listdir(FRAMES_DIR):
        if f.endswith('.jpg'):
            os.remove(os.path.join(FRAMES_DIR, f))
    for f in os.listdir(FRAME_STATIC):
        if f.endswith('.jpg'):
            os.remove(os.path.join(FRAME_STATIC, f))

    cap        = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval   = max(1, int(native_fps / fps_target))
    count = saved = 0

    set_status("Extraction des frames", 0, max(1, total_fr // interval))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            fname = f"frame_{saved}.jpg"
            cv2.imwrite(os.path.join(FRAMES_DIR, fname), frame)
            # Copie dans static pour affichage web
            small = cv2.resize(frame, (320, 180))
            cv2.imwrite(os.path.join(FRAME_STATIC, fname), small)
            saved += 1
            set_status("Extraction des frames", saved,
                       max(1, total_fr // interval))
        count += 1

    cap.release()
    return saved

def extract_audio_from_video(video_path):
    """Extrait la piste audio en WAV."""
    wav_path = os.path.join(AUDIOS_DIR, "audio.wav")
    try:
        from moviepy import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None
        clip.audio.write_audiofile(wav_path, logger=None)
        clip.close()
        return wav_path
    except Exception as e:
        print(f"[extract_audio] Erreur : {e}")
        return None

def transcribe_audio(wav_path):
    """Transcription audio → texte via Google Speech API."""
    try:
        import speech_recognition as sr
        r       = sr.Recognizer()
        # Essai en anglais d'abord, puis français
        for lang in ["en-US", "fr-FR", "ar-SA"]:
            try:
                with sr.AudioFile(wav_path) as src:
                    r.adjust_for_ambient_noise(src, duration=0.3)
                    audio = r.record(src)
                text = r.recognize_google(audio, language=lang)
                if text:
                    with open(TRANSCRIPT, "w", encoding="utf-8") as f:
                        f.write(text)
                    return text
            except sr.UnknownValueError:
                continue
            except Exception:
                continue
    except Exception as e:
        print(f"[transcribe] Erreur : {e}")
    return ""

def build_histograms(n_frames):
    """Calcule les histogrammes RGB pour toutes les frames."""
    # Nettoyer histogrammes précédents
    for f in os.listdir(HIST_STATIC):
        if f.endswith('.png'):
            os.remove(os.path.join(HIST_STATIC, f))

    cache = {}
    files = sorted(
        [f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')],
        key=lambda x: int(x.replace('frame_','').replace('.jpg',''))
    )

    set_status("Calcul des histogrammes", 0, len(files))

    for i, fname in enumerate(files):
        fid = fname.replace('frame_','').replace('.jpg','')
        img = cv2.imread(os.path.join(FRAMES_DIR, fname))
        if img is None:
            continue

        # Vecteur histogramme (192 dimensions)
        vec = []
        hists_plot = []
        for ch in range(3):
            h = cv2.calcHist([img], [ch], None, [64], [0, 256])
            h = cv2.normalize(h, h).flatten()
            vec.extend(h.tolist())
            hists_plot.append(h)
        cache[fid] = vec

        # Générer image histogramme
        fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0d0e11")
        ax.set_facecolor("#161720")
        colors = ["#4287f5", "#34d399", "#f87171"]
        labels = ["Bleu", "Vert", "Rouge"]
        x = np.arange(64)
        for j, (h, c, l) in enumerate(zip(hists_plot, colors, labels)):
            ax.plot(x, h, color=c, lw=1.4, label=l, alpha=0.9)
            ax.fill_between(x, h.flatten(), alpha=0.1, color=c)
        ax.set_title(f"Frame {fid}", color="#e8eaf0", fontsize=8)
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#1e2030", edgecolor="#333", labelcolor="#aaa")
        ax.tick_params(colors="#444", labelsize=6)
        for sp in ax.spines.values(): sp.set_color("#333")
        plt.tight_layout(pad=0.4)
        plt.savefig(os.path.join(HIST_STATIC, f"hist_{fid}.png"),
                    dpi=72, facecolor=fig.get_facecolor())
        plt.close()

        set_status("Calcul des histogrammes", i + 1, len(files))

    with open(HIST_CACHE, "w") as f:
        json.dump(cache, f)

    return cache

def build_inverted_index(transcription):
    """Construit l'index inversé depuis la transcription."""
    if not transcription.strip():
        return {}
    tokens = normalize_text(transcription)
    freq   = Counter(tokens)
    index  = {}
    for pos, word in enumerate(tokens):
        if word not in index:
            index[word] = {"positions": [], "freq": 0}
        index[word]["positions"].append(pos)
        index[word]["freq"] = freq[word]
    # Tri par fréquence
    index = dict(sorted(index.items(), key=lambda x: x[1]["freq"], reverse=True))
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index

def detect_keyframes(threshold=3.0):
    """Détecte les keyframes par différence inter-frames."""
    for f in os.listdir(KEYFRAMES_DIR):
        os.remove(os.path.join(KEYFRAMES_DIR, f))

    files = sorted(
        [f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')],
        key=lambda x: int(x.replace('frame_','').replace('.jpg',''))
    )
    prev = None
    kfs  = []
    saved = 0

    for fname in files:
        fid   = int(fname.replace('frame_','').replace('.jpg',''))
        frame = cv2.imread(os.path.join(FRAMES_DIR, fname))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff  = cv2.absdiff(prev, gray)
            score = diff.mean()
            if score > threshold:
                kf_name = f"keyframe_{saved}.jpg"
                cv2.imwrite(os.path.join(KEYFRAMES_DIR, kf_name), frame)
                kfs.append({"id": saved, "frame_id": fid,
                            "file": kf_name, "diff": round(float(score), 2)})
                saved += 1
        prev = gray

    kf_json = os.path.join(DATA_DIR, "keyframes.json")
    with open(kf_json, "w") as f:
        json.dump(kfs, f, indent=2)
    return kfs


def run_pipeline(video_path, video_name):
    """Pipeline complet dans un thread séparé."""
    global pipeline_status
    pipeline_status.update({"running": True, "done": False,
                             "error": None, "video_name": video_name})
    try:
        # 1) Extraction frames
        set_status("📽  Extraction des frames...")
        n_frames = extract_frames_from_video(video_path, fps_target=1)
        print(f"[pipeline] {n_frames} frames extraites")

        # 2) Détection keyframes
        set_status("🔑  Détection des keyframes...")
        kfs = detect_keyframes(threshold=3.0)
        print(f"[pipeline] {len(kfs)} keyframes détectées")

        # 3) Extraction audio
        set_status("🎵  Extraction audio...")
        wav = extract_audio_from_video(video_path)

        # 4) Transcription
        transcription = ""
        if wav:
            set_status("🎤  Transcription audio → texte...")
            transcription = transcribe_audio(wav)
            print(f"[pipeline] Transcription : {transcription[:80]}...")
        else:
            print("[pipeline] Pas d'audio détecté")

        # 5) Histogrammes
        set_status("🎨  Calcul des histogrammes RGB...")
        hist_cache = build_histograms(n_frames)
        print(f"[pipeline] {len(hist_cache)} histogrammes calculés")

        # 6) Index inversé
        set_status("📚  Construction de l'index inversé...")
        index = build_inverted_index(transcription)
        print(f"[pipeline] Index : {len(index)} mots")

        # 7) Détection d'objets (NOUVEAU)
        set_status("🔎  Détection d'objets dans les frames...")
        obj_index = build_object_index(FRAMES_DIR, OBJ_INDEX)
        n_objects = len(obj_index.get("by_object", {}))
        print(f"[pipeline] Objets détectés : {list(obj_index.get('by_object',{}).keys())}")

        pipeline_status.update({
            "running": False, "done": True, "step": "✅ Indexation terminée !",
            "progress": 100, "total": 100,
            "n_frames": n_frames, "n_kf": len(kfs),
            "n_words": len(index), "n_objects": n_objects,
            "transcription": transcription
        })

    except Exception as e:
        import traceback
        pipeline_status.update({
            "running": False, "done": False,
            "error": str(e), "step": f"❌ Erreur : {e}"
        })
        print(f"[pipeline] ERREUR : {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────
#  CHARGEMENT INDEX AU DÉMARRAGE
# ─────────────────────────────────────────────────────────────

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_hist_cache():
    if os.path.exists(HIST_CACHE):
        with open(HIST_CACHE, "r") as f:
            return json.load(f)
    return {}

def load_transcription():
    if os.path.exists(TRANSCRIPT):
        with open(TRANSCRIPT, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def load_object_index():
    if os.path.exists(OBJ_INDEX):
        with open(OBJ_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"by_frame": {}, "by_object": {}}


# ─────────────────────────────────────────────────────────────
#  ROUTES FLASK
# ─────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

# ── Upload vidéo ──────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload_video():
    """Upload + lancement automatique du pipeline."""
    if "video" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Nom de fichier vide"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
        return jsonify({"error": f"Format non supporté : {ext}"}), 400

    if pipeline_status["running"]:
        return jsonify({"error": "Un traitement est déjà en cours"}), 409

    # Sauvegarder
    safe_name = f"video_{int(time.time())}{ext}"
    save_path = os.path.join(VIDEOS_DIR, safe_name)
    f.save(save_path)

    # Lancer le pipeline dans un thread
    t = threading.Thread(target=run_pipeline,
                         args=(save_path, f.filename), daemon=True)
    t.start()

    return jsonify({"ok": True, "filename": f.filename})

# ── Statut du pipeline ────────────────────────────────────

@app.route("/pipeline/status")
def get_pipeline_status():
    return jsonify(pipeline_status)

# ── Recherche textuelle ───────────────────────────────────

@app.route("/search/text")
def search_text():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify({"results": [], "query": "", "total": 0})

    index        = load_index()
    transcription = load_transcription()
    words        = query.split()
    scores       = {}

    for word in words:
        for key, data in index.items():
            if word in key or key in word:
                for pos in data["positions"]:
                    scores[pos] = scores.get(pos, 0) + data["freq"]

    sorted_r  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_score = sorted_r[0][1] if sorted_r else 1
    trans_words = transcription.split()

    results = []
    for pos, score in sorted_r:
        start   = max(0, pos - 5)
        end     = min(len(trans_words), pos + 6)
        snippet = " ".join(trans_words[start:end])
        results.append({
            "position":  pos,
            "score":     score,
            "score_pct": round((score / max_score) * 100),
            "frame":     f"frame_{pos}.jpg",
            "hist":      f"hist_{pos}.png",
            "snippet":   snippet
        })

    return jsonify({
        "results":       results,
        "query":         query,
        "total":         len(results),
        "transcription": transcription
    })

# ── Recherche par image ───────────────────────────────────

@app.route("/search/image", methods=["POST"])
def search_by_image():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image"}), 400

    f = request.files["image"]
    tmp = os.path.join(DATA_DIR, "query_tmp.jpg")
    f.save(tmp)

    img = cv2.imread(tmp)
    if img is None:
        return jsonify({"error": "Image illisible"}), 400

    # Histogramme de la requête
    q_vec = []
    for ch in range(3):
        h = cv2.calcHist([img], [ch], None, [64], [0, 256])
        h = cv2.normalize(h, h).flatten()
        q_vec.extend(h.tolist())
    q_arr = np.array(q_vec, dtype=np.float32)

    hist_cache = load_hist_cache()
    if not hist_cache:
        return jsonify({"error": "Aucun histogramme indexé. Uploadez d'abord une vidéo."}), 400

    distances = []
    for fid, vec in hist_cache.items():
        v = np.array(vec, dtype=np.float32)
        corr = cv2.compareHist(q_arr, v, cv2.HISTCMP_CORREL)
        dist = float(1.0 - ((corr + 1) / 2))
        distances.append((fid, dist))

    distances.sort(key=lambda x: x[1])

    results = []
    for fid, dist in distances[:20]:
        sim = round((1 - dist) * 100)
        results.append({
            "frame_id":       fid,
            "frame":          f"frame_{fid}.jpg",
            "hist":           f"hist_{fid}.png",
            "distance":       round(dist, 4),
            "similarity_pct": sim
        })

    return jsonify({
        "results": results,
        "total":   len(results),
        "method":  "Histogramme RGB · 64 bins/canal · Corrélation"
    })

# ── API info ──────────────────────────────────────────────

@app.route("/search/objects")
def search_objects_route():
    """
    Recherche par objets visuels détectés dans les frames.
    Cherche : soleil, ciel, herbe, eau, feu, personne, chat, cercle...
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": [], "query": "", "total": 0})

    obj_index      = load_object_index()
    by_frame       = obj_index.get("by_frame", {})
    matched        = search_objects(obj_index, query)
    max_score      = matched[0][1] if matched else 1
    transcription  = load_transcription()

    results = []
    for fid, score in matched:
        frame_data = by_frame.get(str(fid), {})
        results.append({
            "frame_id":   int(fid),
            "score":      score,
            "score_pct":  round((score / max_score) * 100),
            "objects":    frame_data.get("objects", []),
            "labels_fr":  frame_data.get("labels_fr", []),
            "details":    frame_data.get("details", []),
            "frame":      f"frame_{fid}.jpg",
            "hist":       f"hist_{fid}.png",
        })

    return jsonify({
        "results":      results,
        "query":        query,
        "total":        len(results),
        "all_objects":  list(obj_index.get("by_object", {}).keys()),
        "method":       "Détection hybride : Haar Cascade + HSV + Contours"
    })

@app.route("/search/image/objects", methods=["POST"])
def search_image_objects():
    """
    Recherche par image : détecte les objets dans l image requête
    puis cherche les frames qui contiennent les mêmes objets.
    """
    if "image" not in request.files:
        return jsonify({"error": "Aucune image"}), 400

    f   = request.files["image"]
    tmp = os.path.join(DATA_DIR, "query_obj_tmp.jpg")
    f.save(tmp)

    img = cv2.imread(tmp)
    if img is None:
        return jsonify({"error": "Image illisible"}), 400

    # Détecter les objets dans l image requête
    detected    = detect_objects(img)
    query_words = " ".join(detected["objects"])
    labels_fr   = detected["labels_fr"]

    if not query_words:
        return jsonify({
            "error": "Aucun objet détecté dans l image requête",
            "results": []
        })

    # Chercher dans l index
    obj_index = load_object_index()
    by_frame  = obj_index.get("by_frame", {})
    matched   = search_objects(obj_index, query_words)
    max_score = matched[0][1] if matched else 1

    results = []
    for fid, score in matched:
        frame_data = by_frame.get(str(fid), {})
        results.append({
            "frame_id":  int(fid),
            "score":     score,
            "score_pct": round((score / max_score) * 100),
            "objects":   frame_data.get("objects", []),
            "labels_fr": frame_data.get("labels_fr", []),
            "frame":     f"frame_{fid}.jpg",
        })

    return jsonify({
        "results":         results[:20],
        "total":           len(results),
        "query_objects":   detected["objects"],
        "query_labels_fr": labels_fr,
        "method":          "Détection objets dans image requête + index"
    })

@app.route("/api/objects")
def api_objects():
    """Liste tous les objets détectés dans la vidéo indexée."""
    obj_index = load_object_index()
    by_object = obj_index.get("by_object", {})
    summary   = {obj: len(fids) for obj, fids in by_object.items()}
    return jsonify({
        "objects":  summary,
        "total":    len(by_object),
        "keywords": list(SEARCH_KEYWORDS.keys())
    })

@app.route("/api/info")
def api_info():
    index        = load_index()
    hist_cache   = load_hist_cache()
    transcription = load_transcription()
    obj_index    = load_object_index()

    kf_file = os.path.join(DATA_DIR, "keyframes.json")
    kfs = []
    if os.path.exists(kf_file):
        with open(kf_file) as f:
            kfs = json.load(f)

    return jsonify({
        "indexed":         os.path.exists(INDEX_FILE),
        "n_words":         len(index),
        "n_frames":        len(hist_cache),
        "n_keyframes":     len(kfs),
        "n_objects":       len(obj_index.get("by_object", {})),
        "transcription":   transcription,
        "words":           list(index.keys())[:30],
        "objects":         list(obj_index.get("by_object", {}).keys()),
        "video_name":      pipeline_status.get("video_name", "")
    })

# ── Fichiers statiques dynamiques ────────────────────────

@app.route("/static/frames/<filename>")
def serve_frame(filename):
    return send_from_directory(FRAME_STATIC, filename)

@app.route("/static/histograms/<filename>")
def serve_hist(filename):
    return send_from_directory(HIST_STATIC, filename)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ARYLIN — Moteur de Recherche Multimédia")
    print("  Version Dynamique — Upload n'importe quelle vidéo")
    print("=" * 55)
    print("  → http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, threaded=True)
