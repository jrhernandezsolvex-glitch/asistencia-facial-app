import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import cv2
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Asistencia Facial", page_icon="✅", layout="centered")

APP_TITLE = "✅ Asistencia por Selfie"
DEFAULT_THRESHOLD = 0.38

SHEET_NAME = st.secrets.get("SHEET_NAME", "Asistencia Facial")
WORKSHEET_NAME = st.secrets.get("WORKSHEET_NAME", "Hoja 1")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")  # opcional

DB_FILE = "face_db_multi.npz"

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

model = load_model()

def get_face_embedding(img_bgr):
    faces = model.get(img_bgr)
    if not faces:
        return None
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    return faces[0].normed_embedding.astype(np.float32)

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    data = np.load(DB_FILE, allow_pickle=True)
    names = data["names"].tolist()
    embs  = data["embs"]
    db = {}
    for name, emb in zip(names, embs):
        db.setdefault(name, []).append(emb.astype(np.float32))
    return db

def save_db(db):
    names, embs = [], []
    for name, arr in db.items():
        for emb in arr:
            names.append(name)
            embs.append(emb)
    if not embs:
        np.savez(DB_FILE, names=np.array([], dtype=object), embs=np.zeros((0,512), np.float32))
    else:
        np.savez(DB_FILE, names=np.array(names, dtype=object), embs=np.stack(embs).astype(np.float32))

def cosine_sim(a, b):
    return float(np.dot(a, b))

def identify(db, emb, threshold):
    best_name, best_score = None, -1.0
    for name, emb_list in db.items():
        score = max(cosine_sim(emb, ref) for ref in emb_list)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < threshold:
        return None, best_score
    return best_name, best_score

@st.cache_resource
def get_gs_client():
    creds_info = st.secrets.get("gcp_service_account")
    if creds_info is None:
        st.error("Faltan credenciales en Secrets: gcp_service_account")
        st.stop()
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(dict(creds_info), scopes=scopes)
    return gspread.authorize(creds)

def open_sheet():
    gc = get_gs_client()
    sh = gc.open(SHEET_NAME)
    return sh.worksheet(WORKSHEET_NAME)

def read_attendance_df(ws):
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame(columns=["fecha_hora","nombre","score"])
    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    for col in ["fecha_hora","nombre","score"]:
        if col not in df.columns:
            df[col] = ""
    return df[["fecha_hora","nombre","score"]]

def already_marked_today(df, name):
    today = date.today().isoformat()
    if df.empty:
        return False
    return any((df["nombre"].astype(str) == name) &
               (df["fecha_hora"].astype(str).str.startswith(today)))

def append_attendance(ws, name, score):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws.append_row([now, name, round(float(score), 4)], value_input_option="USER_ENTERED")

st.title(APP_TITLE)
st.caption("Toma una selfie y se registrará tu asistencia en Google Sheets.")

db = load_db()

tab1, tab2 = st.tabs(["📸 Marcar asistencia", "⚙️ Administración"])

with tab1:
    threshold = st.slider("Umbral de reconocimiento", 0.30, 0.60, float(DEFAULT_THRESHOLD), 0.01)
    st.write(f"Personas registradas: **{len(db)}**")

    img_file = st.camera_input("Toma tu selfie")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        emb = get_face_embedding(img_bgr)
        if emb is None:
            st.error("No se detectó rostro. Intenta con mejor luz y más cerca.")
        elif len(db) == 0:
            st.error("❌ No hay personas registradas.")
        else:
            name, score = identify(db, emb, threshold)
            if name is None:
                st.error(f"No identificado. score={score:.3f} (umbral={threshold:.2f})")
            else:
                ws = open_sheet()
                df = read_attendance_df(ws)
                if already_marked_today(df, name):
                    st.info(f"{name} ya marcó asistencia hoy. (no se duplica)")
                else:
                    append_attendance(ws, name, score)
                    st.success(f"Asistencia registrada: {name} (score={score:.3f})")

                st.subheader("Últimos registros")
                st.dataframe(read_attendance_df(ws).tail(10), use_container_width=True)

with tab2:
    st.subheader("Enrolar personas (solo admin)")

    if ADMIN_PASSWORD:
        pwd = st.text_input("Contraseña admin", type="password")
        if pwd != ADMIN_PASSWORD:
            st.info("Ingresa la contraseña para enrolar.")
            st.stop()

    st.write("Recomendado: **3 selfies** (frente, leve izq, leve der).")

    new_name = st.text_input("Nombre (exacto)")
    n_photos = st.selectbox("Cantidad de selfies", [1,2,3,4,5], index=2)

    if "enroll_embs" not in st.session_state:
        st.session_state.enroll_embs = []

    enroll_img = st.camera_input("Selfie para enrolamiento", key="enroll_cam")

    if enroll_img is not None:
        b = np.asarray(bytearray(enroll_img.getvalue()), dtype=np.uint8)
        bgr = cv2.imdecode(b, cv2.IMREAD_COLOR)
        emb = get_face_embedding(bgr)
        if emb is None:
            st.error("No se detectó rostro en esta selfie.")
        else:
            st.session_state.enroll_embs.append(emb)
            st.success(f"Selfie guardada ({len(st.session_state.enroll_embs)}/{int(n_photos)})")

    if st.button("✅ Guardar enrolamiento"):
        if not new_name.strip():
            st.error("Escribe un nombre.")
        elif len(st.session_state.enroll_embs) < int(n_photos):
            st.error("Aún faltan selfies para completar el enrolamiento.")
        else:
            name = new_name.strip()
            db.setdefault(name, [])
            db[name].extend(st.session_state.enroll_embs[:int(n_photos)])
            save_db(db)
            st.session_state.enroll_embs = []
            st.success(f"Registrado: {name}")
            st.rerun()

    st.divider()
    st.subheader("Personas registradas")
    st.write({k: len(v) for k,v in db.items()})

    del_name = st.text_input("Nombre exacto a eliminar")
    if st.button("🗑️ Eliminar persona"):
        if del_name.strip() in db:
            del db[del_name.strip()]
            save_db(db)
            st.success("Eliminado.")
            st.rerun()
        else:
            st.error("No existe ese nombre.")
