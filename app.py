import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image
import io
import time
import os

# --- SYSTÈME DE MOT DE PASSE ---
# (Ton code de mot de passe ici...)

# --- FIX OPENCV FOR STREAMLIT CLOUD ---
# Force la désinstallation de la version graphique qui fait planter, et installe la version serveur
try:
    import cv2
    if "headless" not in cv2.__file__:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
# --- END FIX ---

import numpy as np
import pandas as pd
# ... la suite de ton code (import plotly, etc.)

# --- SYSTÈME DE MOT DE PASSE ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets.get("password", "alstom2026"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Ne pas stocker le mot de passe
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Mot de passe fourni par le candidat :", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Mot de passe fourni par le candidat :", type="password", on_change=password_entered, key="password")
        st.error("😕 Mot de passe incorrect. Veuillez contacter Ibrahima DIALLO.")
        st.stop()

check_password()
# --- FIN DU SYSTÈME DE MOT DE PASSE --- 

# ==========================================
# CONFIGURATION & CONSTANTES
# ==========================================
TT_SLICE_W = 640
TT_IMG_W = 640
TT_IMG_H = 640
TT_SEP_GREEN = 55

FILTER_MIN = 900    # min length in mm
FILTER_MAX = 4000   # max length in mm

st.set_page_config(page_title="Détection Compteurs d'Essieux", page_icon="🚄", layout="wide")
st.title("🚄 Détection de Compteurs d'Essieux (Modèle 3 Slices Denses)")
st.markdown("Pipeline : **Compression x6** ➡️ **TT 3 Slices** ➡️ **YOLO** ➡️ **NMS 1D Itératif**")

# ==========================================
# FONCTIONS DU PIPELINE (VOTRE CODE EXACT)
# ==========================================

def compress_signal(signal, factor=6):
    if factor == 1: return signal
    N, n_ch = signal.shape
    new_N = N // factor
    compressed = signal[:new_N*factor].reshape(new_N, factor, n_ch).mean(axis=1)
    return compressed

def calculate_all_thresholds(signal):
    # signal shape: (N, 15)
    thresholds = []
    for ch in range(signal.shape[1]):
        data = signal[:, ch]
        mu = np.mean(data)
        sigma = np.std(data)
        for _ in range(5):
            mask = np.abs(data - mu) < 3 * sigma
            if np.sum(mask) < 0.3 * len(data): break
            mu = np.mean(data[mask])
            sigma = np.std(data[mask])
        thresholds.append(3 * sigma)
    return thresholds

def threshold_transform(signal_window, thresholds, img_width=TT_SLICE_W):
    n_samples, n_ch = signal_window.shape
    red  = np.zeros((n_ch, img_width), dtype=np.uint8)
    blue = np.zeros((n_ch, img_width), dtype=np.uint8)
    comp = n_samples / img_width

    for col in range(img_width):
        s0 = int(col * comp)
        s1 = min(int((col+1) * comp), n_samples)
        if s0 >= n_samples: break
        for ch in range(n_ch):
            amp = float(np.mean(signal_window[s0:s1, ch])) if s1 > s0 else float(signal_window[s0, ch])
            th = thresholds[ch]
            if isinstance(th, (tuple, list)) and len(th) == 2:
                mu, lv = th
            else:
                mu, lv = 0.0, float(th)
            v = abs(amp)
            if lv == 0:    intensity = 0
            elif v < lv:   intensity = int(v / lv * 64)
            else:          intensity = min(int(np.log10(v / lv) * 191 + 64), 255)
            intensity = max(0, intensity)
            if amp >= 0: red[ch, col]  = max(red[ch, col],  intensity)
            else:        blue[ch, col] = max(blue[ch, col], intensity)

    img = np.zeros((n_ch, img_width, 3), dtype=np.uint8)
    img[:, :, 0] = red
    img[:, :, 2] = blue
    return img

def build_tt_640x640(defectogram, thresholds, slice_w=TT_SLICE_W, img_w=TT_IMG_W, img_h=TT_IMG_H):
    N, n_ch  = len(defectogram), defectogram.shape[1]
    n_slices = int(np.ceil(N / slice_w))
    row_h    = n_ch + 1
    canvas_h = n_slices * row_h - 1   
    
    canvas   = np.zeros((canvas_h, slice_w, 3), dtype=np.uint8)
    meta     = []

    for k in range(n_slices):
        s0, s1 = k * slice_w, min((k+1) * slice_w, N)
        seg    = defectogram[s0:s1, :]
        if len(seg) < slice_w:
            pad = np.zeros((slice_w - len(seg), n_ch), dtype=seg.dtype)
            seg = np.vstack([seg, pad])
            
        img_slice = threshold_transform(seg, thresholds, img_width=slice_w)
        y0, y1   = k * row_h, k * row_h + n_ch
        canvas[y0:y1, :, :] = img_slice
        if k < n_slices - 1:
            canvas[y1, :, 1] = TT_SEP_GREEN
            
        meta.append({'k': k, 'samp_s': s0, 'samp_e': s1, 'y0_cv': y0, 'y1_cv': y1})

    img_640 = cv2.resize(canvas, (img_w, img_h), interpolation=cv2.INTER_AREA)
    scale_y = img_h / canvas_h
    return img_640, meta, canvas_h, scale_y

def nms_1d_iteratif(detections, min_overlap_ratio=0.20):
    if not detections: return []
    dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    final = []
    while dets:
        best = dets.pop(0)
        changed = True
        while changed:
            changed = False
            new_dets = []
            for d in dets:
                overlap = min(best['end'], d['end']) - max(best['start'], d['start'])
                len_min = min(best['end'] - best['start'], d['end'] - d['start'])
                if overlap > 0 and (overlap / len_min) > min_overlap_ratio:
                    best['start'] = min(best['start'], d['start'])
                    best['end'] = max(best['end'], d['end'])
                    best['confidence'] = max(best['confidence'], d['confidence'])
                    changed = True
                else:
                    new_dets.append(d)
            dets = new_dets
        final.append(best)
    return final

def detect_axle_counters_dense_app(model, defectogram, compress_factor=6, confidence_threshold=0.65):
    thresholds = calculate_all_thresholds(defectogram)
    defecto_c = compress_signal(defectogram, compress_factor)
    
    WINDOW_SIZE_C = 1920  # 3 tranches de 640
    STRIDE_C = 1280       
    raw_dets = []
    tt_images_with_boxes = []

    for start_c in range(0, len(defecto_c) - WINDOW_SIZE_C, STRIDE_C):
        end_c = start_c + WINDOW_SIZE_C
        window_c = defecto_c[start_c:end_c, :]
        
        img_tt, meta, canvas_h, scale_y = build_tt_640x640(window_c, thresholds, slice_w=640)
        
        results = model.predict(img_tt, conf=confidence_threshold, verbose=False)
        
        # Sauvegarder l'image annotée pour Streamlit
        if len(results[0].boxes) > 0:
            annotated_frame = results[0].plot() # Dessine les boîtes natives de YOLO
            tt_images_with_boxes.append((start_c * compress_factor, annotated_frame))
            
        for result in results:
            for box in result.boxes:
                xc_n = float(box.xywhn[0][0])
                yc_n = float(box.xywhn[0][1])
                w_n  = float(box.xywhn[0][2])
                conf = float(box.conf[0])
                
                yc_px = yc_n * 640
                k_found = -1
                for sl in meta:
                    yt = int(sl['y0_cv'] * scale_y)
                    yb = int(sl['y1_cv'] * scale_y)
                    if yt <= yc_px <= yb:
                        k_found = sl['k']
                        break
                
                if k_found == -1: continue
                
                x_in_slice_c = xc_n * 640 
                w_in_slice_c = w_n * 640
                slice_offset_c = k_found * 640
                
                det_start_c = int(start_c + slice_offset_c + x_in_slice_c - w_in_slice_c / 2)
                det_end_c   = int(start_c + slice_offset_c + x_in_slice_c + w_in_slice_c / 2)
                
                raw_dets.append({
                    'start'     : det_start_c * compress_factor,
                    'end'       : det_end_c * compress_factor,
                    'confidence': conf,
                })

    if not raw_dets:
        return [], tt_images_with_boxes

    # 1. Filtre longueur
    valid_dets = [d for d in raw_dets if FILTER_MIN <= (d['end'] - d['start']) <= FILTER_MAX]

    # 2. NMS 1D
    valid_dets = nms_1d_iteratif(valid_dets)

    # 3. Extension 5%
    def extend_detection_smart(det, extension_percent=0.05):
        length = det['end'] - det['start']
        center = (det['start'] + det['end']) / 2
        extension = length * extension_percent
        det['start'] = max(0, int(center - length/2 - extension))
        det['end'] = int(center + length/2 + extension)
        return det

    valid_dets = [extend_detection_smart(d, 0.05) for d in valid_dets]

    # 4. 2eme passe NMS
    valid_dets = nms_1d_iteratif(valid_dets)

    # 5. Filtre Zone (Distance min 3m)
    valid_dets = sorted(valid_dets, key=lambda x: x['confidence'], reverse=True)
    MIN_DISTANCE = 3000
    final = [valid_dets[0]] if valid_dets else []
    
    for d in valid_dets[1:]:
        cd = (d['start'] + d['end']) / 2
        keep = True
        for k in final:
            ck = (k['start'] + k['end']) / 2
            if abs(cd - ck) < MIN_DISTANCE:
                keep = False
                break
        if keep:
            final.append(d)

    return final, tt_images_with_boxes

# ==========================================
# INTERFACE STREAMLIT
# ==========================================

@st.cache_resource
def load_model(path):
    if os.path.exists(path): return YOLO(path)
    return None

model = load_model("best.pt")
if model is None:
    st.error("⚠️ Modèle `best.pt` introuvable ! Placez votre modèle entraîné (3 slices) dans le dossier.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Paramètres")
    conf_thresh = st.slider("Seuil de confiance YOLO", 0.0, 1.0, 0.65, 0.05)
    st.info("Modèle : 3 Slices Denses\nCompression : x6\nNMS : 1D Itératif")

st.subheader("📁 Chargement des données")
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    uploaded_data = st.file_uploader("1. Fichier Défectogramme (.txt) - OBLIGATOIRE", type=["txt"])

with col_upload2:
    uploaded_labels = st.file_uploader("2. Fichier Labels / Vérité terrain (.txt) - OPTIONNEL", type=["txt"])

if uploaded_data is not None:
    try:
        stringio = io.StringIO(uploaded_data.getvalue().decode("utf-8"))
        data = np.loadtxt(stringio, dtype=int)
        if data.ndim == 1: data = data.reshape(-1, 1)
        
        # VOTRE CODE ATTEND (N, 15) POUR LE SIGNAL. On s'assure donc de transposer.
        if data.shape[0] < data.shape[1]:
            signal = data.T # Passe de (15, N) à (N, 15)
        else:
            signal = data   # Déjà en (N, 15)
            
        st.success(f"✅ Signal chargé : {signal.shape[0]} mm, {signal.shape[1]} canaux")
        
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        st.stop()

    # Lecture Labels
    ground_truths = []
    if uploaded_labels is not None:
        try:
            stringio_lbl = io.StringIO(uploaded_labels.getvalue().decode("utf-8"))
            lines = stringio_lbl.read().strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_mm, end_mm = int(parts[1]), int(parts[2])
                    ground_truths.append({'start': start_mm, 'end': end_mm})
            st.info(f"✅ Vérité terrain chargée : {len(ground_truths)} compteur(s) réel(s)")
        except Exception as e:
            st.warning(f"Erreur labels ignorés : {e}")

    if st.button("🚀 Lancer la détection", type="primary"):
        with st.spinner("Analyse en cours (3 Slices Denses)..."):
            start_time = time.time()
            
            final_dets, tt_images = detect_axle_counters_dense_app(
                model, signal, compress_factor=6, confidence_threshold=conf_thresh
            )
            
            proc_time = time.time() - start_time

        st.balloons()
        st.header("📊 Résultats de la détection")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("⏱️ Temps", f"{proc_time:.2f} sec")
        col2.metric("🎯 Compteurs finaux", len(final_dets))
        col3.metric("📏 Signal analysé", f"{signal.shape[0]} mm")

        if not final_dets:
            st.warning("Aucun compteur détecté.")
        else:
            # --- Visualisation 1 : Images TT ---
            st.subheader("👁️ Images TT avec prédictions YOLO")
            st.markdown("*Les boîtes sont dessinées automatiquement par YOLO sur l'image 3 Slices.*")
            
            # Afficher les 3 meilleures images
            for pos, img in tt_images[:3]:
                st.image(img, caption=f"Fenêtre démarrant à {pos} mm", use_column_width="auto")

            # --- Visualisation 2 : Graphique 1D ---
            st.subheader("📈 Localisation sur le signal 1D réel")
            
            channel_to_plot = 0
            x_axis = np.arange(signal.shape[0])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_axis, y=signal[:, channel_to_plot], mode='lines', name='Canal 1',
                line=dict(color='blue', width=1)
            ))
            
            if ground_truths:
                for i, gt in enumerate(ground_truths):
                    fig.add_vrect(x0=gt['start'], x1=gt['end'], fillcolor="green", opacity=0.3, line_width=2, line_color="green", annotation_text=f"Réel {i+1}", annotation_position="top left")
            
            for i, det in enumerate(final_dets):
                fig.add_vrect(x0=det['start'], x1=det['end'], fillcolor="yellow", opacity=0.4, line_width=3, line_color="orange", annotation_text=f"Prédit {i+1} ({det['confidence']:.0%})", annotation_position="bottom left")
            
            fig.update_layout(title="Signal CDF : Prédictions (Jaune) vs Réalité (Vert)", xaxis_title="Position rail (mm)", yaxis_title="Amplitude", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # --- Tableau ---
            st.subheader("📋 Détails des détections")
            df = pd.DataFrame([{"Début (mm)": d['start'], "Fin (mm)": d['end'], "Longueur (mm)": d['end']-d['start'], "Confiance": f"{d['confidence']:.2%}"} for d in final_dets])
            st.dataframe(df, use_container_width=True)
