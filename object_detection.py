"""
object_detection.py — Détection d'Objets avec YOLOv8
=====================================================
Utilise YOLOv8 nano pour détecter les vrais objets dans les frames.
Détecte 80 classes COCO : chien, chat, personne, voiture, oiseau...

INSTALLATION :
    pip install ultralytics
    Télécharger yolov8n.pt :
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
    Placer yolov8n.pt dans le dossier arylin_v2/
"""

import cv2, os, json, numpy as np
from collections import defaultdict

YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
_yolo_model     = None

# ── Traductions COCO → Français ──────────────────
COCO_FR = {
    "person":"personne","bicycle":"vélo","car":"voiture","motorcycle":"moto",
    "airplane":"avion","bus":"bus","train":"train","truck":"camion",
    "boat":"bateau","traffic light":"feu de circulation","bench":"banc",
    "bird":"oiseau","cat":"chat","dog":"chien","horse":"cheval",
    "sheep":"mouton","cow":"vache","elephant":"éléphant","bear":"ours",
    "zebra":"zèbre","giraffe":"girafe","backpack":"sac à dos",
    "umbrella":"parapluie","handbag":"sac à main","suitcase":"valise",
    "frisbee":"frisbee","skis":"skis","snowboard":"snowboard",
    "sports ball":"ballon","kite":"cerf-volant","bottle":"bouteille",
    "wine glass":"verre","cup":"tasse","fork":"fourchette","knife":"couteau",
    "spoon":"cuillère","bowl":"bol","banana":"banane","apple":"pomme",
    "sandwich":"sandwich","orange":"orange","broccoli":"brocoli",
    "carrot":"carotte","hot dog":"hot dog","pizza":"pizza","donut":"donut",
    "cake":"gâteau","chair":"chaise","couch":"canapé",
    "potted plant":"plante","bed":"lit","dining table":"table",
    "toilet":"toilettes","tv":"télévision","laptop":"ordinateur portable",
    "mouse":"souris","remote":"télécommande","keyboard":"clavier",
    "cell phone":"téléphone","microwave":"micro-ondes","oven":"four",
    "sink":"évier","refrigerator":"réfrigérateur","book":"livre",
    "clock":"horloge","vase":"vase","scissors":"ciseaux",
    "teddy bear":"ours en peluche","toothbrush":"brosse à dents",
}

# ── Mots-clés recherche (FR/AR → EN COCO) ────────
SEARCH_KEYWORDS = {
    # Français
    "personne":["person"],"homme":["person"],"femme":["person"],"enfant":["person"],
    "gens":["person"],"chien":["dog"],"chat":["cat"],"cheval":["horse"],
    "oiseau":["bird"],"vache":["cow"],"mouton":["sheep"],"éléphant":["elephant"],
    "ours":["bear"],"zèbre":["zebra"],"girafe":["giraffe"],
    "voiture":["car"],"auto":["car"],"moto":["motorcycle"],"vélo":["bicycle"],
    "bus":["bus"],"train":["train"],"avion":["airplane"],"bateau":["boat"],
    "camion":["truck"],"téléphone":["cell phone"],"portable":["cell phone","laptop"],
    "ordinateur":["laptop"],"télévision":["tv"],"livre":["book"],
    "bouteille":["bottle"],"chaise":["chair"],"table":["dining table"],
    "plante":["potted plant"],"pizza":["pizza"],"ballon":["sports ball"],
    "horloge":["clock"],"sac":["backpack","handbag","suitcase"],
    "banane":["banana"],"pomme":["apple"],"gâteau":["cake"],
    "verre":["wine glass","cup"],"fourchette":["fork"],"couteau":["knife"],
    # Anglais direct
    "dog":["dog"],"cat":["cat"],"person":["person"],"car":["car"],
    "bird":["bird"],"horse":["horse"],"elephant":["elephant"],"bear":["bear"],
    "phone":["cell phone"],"laptop":["laptop"],"tv":["tv"],"book":["book"],
    "bottle":["bottle"],"chair":["chair"],"pizza":["pizza"],
    "ball":["sports ball"],"clock":["clock"],"tree":["potted plant"],
    # Arabe translittéré
    "kalb":["dog"],"qitta":["cat"],"sayyara":["car"],
    "tayr":["bird"],"hussan":["horse"],"fil":["elephant"],
}


def yolo_available():
    return os.path.exists(YOLO_MODEL_PATH)


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"\n❌ Modèle YOLO non trouvé : {YOLO_MODEL_PATH}\n"
            "   Téléchargez yolov8n.pt (6 MB) :\n"
            "   https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt\n"
            "   Placez le fichier dans le dossier arylin_v2/\n"
        )
    from ultralytics import YOLO
    print("[YOLO] Chargement du modèle yolov8n.pt...")
    _yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"[YOLO] Prêt — {len(_yolo_model.names)} classes COCO")
    return _yolo_model


def detect_objects(img_bgr, conf_threshold=0.35):
    """
    Détecte les objets réels dans une image avec YOLOv8.

    Returns:
        {
          "objects":   ["dog", "cat"],
          "labels_fr": ["chien", "chat"],
          "details":   [{ label, label_fr, confidence, method, bbox }]
        }
    """
    model   = _load_yolo()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, conf=conf_threshold, verbose=False)

    seen = {}
    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            label      = model.names[class_id]
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if label not in seen or confidence > seen[label]["confidence"]:
                seen[label] = {
                    "label":      label,
                    "label_fr":   COCO_FR.get(label, label),
                    "confidence": round(confidence, 3),
                    "method":     "YOLOv8n",
                    "bbox":       [x1, y1, x2, y2]
                }

    details   = sorted(seen.values(), key=lambda x: x["confidence"], reverse=True)
    return {
        "objects":   [d["label"]    for d in details],
        "labels_fr": [d["label_fr"] for d in details],
        "details":   details
    }


def build_object_index(frames_dir, output_path, status_callback=None):
    """
    Analyse toutes les frames avec YOLO et construit l'index d'objets.

    Structure :
    {
      "by_frame":  { "0": { objects, labels_fr, details }, ... }
      "by_object": { "dog": [0, 3, 7], "cat": [0, 5], ... }
    }
    """
    files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.replace('frame_','').replace('.jpg',''))
    )

    by_frame  = {}
    by_object = defaultdict(list)

    print(f"[YOLO] Analyse de {len(files)} frames...")

    for i, fname in enumerate(files):
        fid = fname.replace('frame_','').replace('.jpg','')
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue

        result = detect_objects(img)
        by_frame[fid] = {
            "objects":   result["objects"],
            "labels_fr": result["labels_fr"],
            "details": [{k:v for k,v in d.items() if k != "bbox"}
                        for d in result["details"]]
        }
        for obj in result["objects"]:
            by_object[obj].append(int(fid))

        if status_callback:
            status_callback(i + 1, len(files))

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(files)}] frame_{fid} → {result['objects'] or ['(rien)']}")

    index = {"by_frame": by_frame, "by_object": dict(by_object)}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\n[YOLO] ✅ {len(by_frame)} frames analysées")
    print(f"       Classes détectées : {sorted(by_object.keys())}")
    return index


def search_objects(object_index, query):
    """
    Cherche les frames contenant les objets de la requête.
    Supporte FR, EN et arabe translittéré.

    Returns:
        liste de (frame_id, score) triée par score décroissant
    """
    words  = query.lower().strip().split()
    by_obj = object_index.get("by_object", {})
    scores = defaultdict(int)

    for word in words:
        # 1) Label COCO direct
        if word in by_obj:
            for fid in by_obj[word]: scores[fid] += 3

        # 2) Via dictionnaire de mots-clés FR/AR
        for mapped in SEARCH_KEYWORDS.get(word, []):
            if mapped in by_obj:
                for fid in by_obj[mapped]: scores[fid] += 3

        # 3) Correspondance partielle anglais
        for obj_label, fids in by_obj.items():
            if word in obj_label or obj_label in word:
                for fid in fids: scores[fid] += 1

        # 4) Correspondance dans traduction française
        for obj_label, fids in by_obj.items():
            fr = COCO_FR.get(obj_label, "")
            if word in fr or fr in word:
                for fid in fids: scores[fid] += 2

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    import sys
    if not yolo_available():
        print("❌ yolov8n.pt non trouvé.")
        print("   https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt")
        sys.exit(1)

    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path:
        print("Usage : python object_detection.py <image.jpg>")
        sys.exit(0)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Impossible de lire : {img_path}")
        sys.exit(1)

    result = detect_objects(img)
    print(f"\n{'Objet':<20} {'Français':<22} {'Confiance':>10}")
    print("─" * 55)
    for d in result["details"]:
        print(f"{d['label']:<20} {d['label_fr']:<22} {d['confidence']*100:>9.1f}%")
