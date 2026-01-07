# app.py — Asistencia facial (selfie 1 a 1) + Enrolamiento (admin) + ENTRADA/SALIDA por horario
#        + Google Sheets (asistencias) + FaceDB en Sheets (embeddings persistentes) + GPS (lat/lon)
#        + Navegación estable (NO vuelve al inicio en cada rerun) usando st.sidebar.radio
# -----------------------------------------------------------------------------------------------
# requirements.txt:
# streamlit
# numpy
# pandas
# opencv-python-headless
# insightface
# onnxruntime
# gspread
# google-auth
# streamlit-js-eval
#
# Streamlit Secrets (Settings → Secrets):
# SHEET_NAME = "Asistencia Facial"
# WORKSHEET_NAME = "Hoja 1"
# SPREADSHEET_ID = "TU_ID_DE_SHEET"   # recomendado (entre /d/ y /edit)
# ADMIN_PASSWORD = "1234"            # opcional
#
# [gcp_service_account]
# type="service_account"
# project_id="..."
# private_key_id="..."
# private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# client_email="...@....iam.gserviceaccount.com"
# client_id="..."
# token_uri="https://oauth2.googleapis.com/token"

import base64
from datetime import datetime, time

import numpy as np
import pandas as pd
import cv2
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

from insightface.app import FaceAnalysis
from zoneinfo import ZoneInfo

from streamlit_js_eval import get_geolocation

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Asistencia Facial", page_icon="✅", layout="centered")

TZ = ZoneInfo("America/Lima")

APP_TITLE = "✅ Asistencia por Selfie (Entrada/Salida)"
DEFAULT_THRESHOLD = 0.38

SHEET_NAME = st.secrets.get("SHEET_NAME", "Asistencia Facial")
WORKSHEET_NAME = st.secrets.get("WORKSHEET_NAME", "Hoja 1")  # asistencia
SPREADSHEET_ID = st.secrets.get("SPREADSHEET_ID", "")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")

FACE_DB_SHEET = "FaceDB"  # hoja para embeddings (persistente)

# Horarios objetivo (informativos)
ENTRY_TARGET = time(7, 45)   # 07:45
EXIT_TARGET  = time(17, 15)  # 17:15

# Ventanas permitidas (tolerancia)
ENTRY_WINDOW_START = time(7, 0)
ENTRY_WINDOW_END   = time(11, 59)

EXIT_WINDOW_START  = time(12, 0)
EXIT_WINDOW_END    = time(20, 30)

# Encabezados
ATT_HEADER = ["fecha_hora", "nombre", "tipo", "score", "lat", "lon", "accuracy_m"]
FACE_HEADER = ["nombre", "emb_b64", "created_at"]

# ----------------------------
# FACE MODEL (cache)
# ----------------------------
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
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    return faces[0].normed_embedding.astype(np.float32)

# ----------------------------
# SIMILARITY / IDENTIFICATION
# ----------------------------
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

# ----------------------------
# GOOGLE SHEETS (service account)
# ----------------------------
@st.cache_resource
def get_gs_client():
    creds_info = st.secrets.get("gcp_service_account")
    if creds_info is None:
        st.error("Faltan credenciales en Secrets: [gcp_service_account].")
        st.stop()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(dict(creds_info), scopes=scopes)
    return gspread.authorize(creds)

def open_spreadsheet():
    gc = get_gs_client()
    if SPREADSHEET_ID and str(SPREADSHEET_ID).strip():
        return gc.open_by_key(str(SPREADSHEET_ID).strip())
    return gc.open(SHEET_NAME)

def open_worksheet(name: str):
    sh = open_spreadsheet()
    try:
        return sh.worksheet(name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=name, rows=2000, cols=20)

# ----------------------------
# SHEET HELPERS (ASISTENCIA)
# ----------------------------
def ensure_attendance_header(ws):
    values = ws.get_all_values()
    if not values:
        ws.append_row(ATT_HEADER, value_input_option="USER_ENTERED")
        return

    header = values[0]

    # Si tu hoja actual tiene 4 columnas, ampliamos a 7 pero SIN insertar filas.
    if header[:4] == ["fecha_hora", "nombre", "tipo", "score"]:
        ws.update("A1:G1", [ATT_HEADER])
        return

    # Si estaba distinto, forzamos la fila 1 a ATT_HEADER (sin insertar)
    ws.update("A1:G1", [ATT_HEADER])

def read_attendance_df(ws):
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame(columns=ATT_HEADER)

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    for col in ATT_HEADER:
        if col not in df.columns:
            df[col] = ""

    return df[ATT_HEADER]

def marked_today(df, name, tipo):
    today_str = datetime.now(TZ).date().isoformat()
    if df.empty:
        return False
    mask = (
        (df["nombre"].astype(str) == name) &
        (df["tipo"].astype(str) == tipo) &
        (df["fecha_hora"].astype(str).str.startswith(today_str))
    )
    return bool(mask.any())

def has_entry_today(df, name):
    return marked_today(df, name, "ENTRADA")

def append_attendance(ws, name, tipo, score, geo):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

    lat, lon, acc = "", "", ""
    if geo and geo.get("coords"):
        lat = geo["coords"].get("latitude", "")
        lon = geo["coords"].get("longitude", "")
        acc = geo["coords"].get("accuracy", "")

    ws.append_row(
        [now, name, tipo, round(float(score), 4), lat, lon, acc],
        value_input_option="USER_ENTERED",
    )

# ----------------------------
# FACE DB in GOOGLE SHEETS (PERSISTENTE)
# ----------------------------
def ensure_face_header(ws):
    values = ws.get_all_values()
    if not values:
        ws.append_row(FACE_HEADER, value_input_option="USER_ENTERED")
        return
    ws.update("A1:C1", [FACE_HEADER])

def emb_to_b64(emb: np.ndarray) -> str:
    raw = emb.astype(np.float32).tobytes()
    return base64.b64encode(raw).decode("utf-8")

def b64_to_emb(s: str):
    try:
        raw = base64.b64decode(s.encode("utf-8"))
        emb = np.frombuffer(raw, dtype=np.float32)
        if emb.shape[0] != 512:
            return None
        return emb
    except Exception:
        return None

@st.cache_data(ttl=30)
def load_db_from_sheet():
    ws = open_worksheet(FACE_DB_SHEET)
    ensure_face_header(ws)

    values = ws.get_all_values()
    if len(values) <= 1:
        return {}

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    db = {}
    for _, r in df.iterrows():
        name = str(r.get("nombre", "")).strip()
        emb_s = str(r.get("emb_b64", "")).strip()
        if not name or not emb_s:
            continue
        emb = b64_to_emb(emb_s)
        if emb is None:
            continue
        db.setdefault(name, []).append(emb.astype(np.float32))
    return db

def add_embeddings_to_sheet(name: str, embs: list):
    ws = open_worksheet(FACE_DB_SHEET)
    ensure_face_header(ws)
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for emb in embs:
        rows.append([name, emb_to_b64(emb), now])

    ws.append_rows(rows, value_input_option="USER_ENTERED")
    load_db_from_sheet.clear()

def delete_person_from_sheet(name: str):
    ws = open_worksheet(FACE_DB_SHEET)
    ensure_face_header(ws)

    values = ws.get_all_values()
    if len(values) <= 1:
        return

    header = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)

    df2 = df[df["nombre"].astype(str).str.strip() != name.strip()].copy()

    ws.clear()
    ws.append_row(FACE_HEADER, value_input_option="USER_ENTERED")
    if not df2.empty:
        ws.append_rows(df2[FACE_HEADER].values.tolist(), value_input_option="USER_ENTERED")

    load_db_from_sheet.clear()

# ----------------------------
# BUSINESS LOGIC: ENTRY / EXIT
# ----------------------------
def decide_tipo(now_t: time):
    if ENTRY_WINDOW_START <= now_t <= ENTRY_WINDOW_END:
        return "ENTRADA"
    if EXIT_WINDOW_START <= now_t <= EXIT_WINDOW_END:
        return "SALIDA"
    return None

# ----------------------------
# UI
# ----------------------------
st.title(APP_TITLE)
st.caption("Toma una selfie. Se registrará ENTRADA o SALIDA según la hora y se guardará en Google Sheets.")

# Navegación estable (evita que se regrese a Marcar asistencia en cada rerun)
if "page" not in st.session_state:
    st.session_state.page = "📸 Marcar asistencia"

page = st.sidebar.radio(
    "Menú",
    ["📸 Marcar asistencia", "⚙️ Administración"],
    index=0 if st.session_state.page == "📸 Marcar asistencia" else 1,
)
st.session_state.page = page

# DB desde Sheets
db = load_db_from_sheet()

# ----------------------------
# PÁGINA 1: MARCAR ASISTENCIA
# ----------------------------
if page == "📸 Marcar asistencia":
    threshold = st.slider("Umbral de reconocimiento", 0.30, 0.60, float(DEFAULT_THRESHOLD), 0.01)
    st.write(f"Personas registradas: **{len(db)}**")

    now_dt = datetime.now(TZ)
    tipo = decide_tipo(now_dt.time())

    st.info(
        f"Hora actual: **{now_dt.strftime('%H:%M:%S')}**  | "
        f"Entrada objetivo: **{ENTRY_TARGET.strftime('%H:%M')}**  | "
        f"Salida objetivo: **{EXIT_TARGET.strftime('%H:%M')}**"
    )
    st.caption(
        f"Ventana ENTRADA: {ENTRY_WINDOW_START.strftime('%H:%M')}–{ENTRY_WINDOW_END.strftime('%H:%M')} | "
        f"Ventana SALIDA: {EXIT_WINDOW_START.strftime('%H:%M')}–{EXIT_WINDOW_END.strftime('%H:%M')}"
    )

    if tipo is None:
        st.warning("Fuera de horario permitido para marcar asistencia.")
    else:
        st.success(f"Tipo de marca detectado: **{tipo}**")

    st.subheader("📍 Geolocalización (GPS)")
    st.caption("Dale permiso al navegador cuando lo pida. Si no lo permites, la marca igual se registra sin GPS.")
    geo = get_geolocation()

    if geo and geo.get("coords"):
        st.success(
            f"GPS OK: lat={geo['coords'].get('latitude')}, lon={geo['coords'].get('longitude')}, "
            f"±{geo['coords'].get('accuracy')} m"
        )
    else:
        st.info("GPS aún no disponible (o permiso denegado).")

    img_file = st.camera_input("Toma tu selfie")

    if img_file is not None:
        raw_bytes = img_file.getvalue()
        file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        emb = get_face_embedding(img_bgr)

        if emb is None:
            st.error("No se detectó rostro. Intenta con mejor luz y más cerca.")
            st.stop()

        if len(db) == 0:
            st.error("❌ No hay personas registradas. Ve a ⚙️ Administración para enrolar.")
            st.stop()

        if tipo is None:
            st.error("No se puede registrar: fuera de la ventana de ENTRADA o SALIDA.")
            st.stop()

        try:
            name, score = identify(db, emb, threshold)

            if name is None:
                st.error(f"No identificado. score={score:.3f} (umbral={threshold:.2f})")
                st.stop()

            ws = open_worksheet(WORKSHEET_NAME)
            ensure_attendance_header(ws)
            df = read_attendance_df(ws)

            # Reglas de negocio
            if tipo == "ENTRADA":
                if marked_today(df, name, "ENTRADA"):
                    st.info(f"{name} ya marcó **ENTRADA** hoy. (no se duplica)")
                    st.stop()

            if tipo == "SALIDA":
                if not has_entry_today(df, name):
                    st.error(f"{name} no tiene **ENTRADA** registrada hoy. No se permite SALIDA.")
                    st.stop()
                if marked_today(df, name, "SALIDA"):
                    st.info(f"{name} ya marcó **SALIDA** hoy. (no se duplica)")
                    st.stop()

            append_attendance(ws, name, tipo, score, geo)

            st.success(f"✅ {tipo} registrada: {name} (score={score:.3f})")

            st.subheader("Últimos registros")
            st.dataframe(read_attendance_df(ws).tail(15), use_container_width=True)

        except Exception as e:
            st.error("Error conectando con Google Sheets (detalle):")
            st.code(f"{type(e).__name__}: {e}")
            st.stop()

# ----------------------------
# PÁGINA 2: ADMINISTRACIÓN
# ----------------------------
if page == "⚙️ Administración":
    st.subheader("Enrolar personas (solo admin)")

    if ADMIN_PASSWORD:
        pwd = st.text_input("Contraseña admin", type="password")
        if pwd != ADMIN_PASSWORD:
            st.info("Ingresa la contraseña para enrolar.")
            st.stop()
    else:
        st.caption("ADMIN_PASSWORD no está configurado. Cualquiera con el link podría enrolar.")

    st.write("Recomendado: **3 selfies** (frente, leve izq, leve der).")

    new_name = st.text_input("Nombre (exacto)")
    n_photos = st.selectbox("Cantidad de selfies", [1, 2, 3, 4, 5], index=2)

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
            embs = st.session_state.enroll_embs[: int(n_photos)]

            add_embeddings_to_sheet(name, embs)

            st.session_state.enroll_embs = []
            st.success(f"Registrado en FaceDB: {name}")

            # Mantenerse en Administración
            st.session_state.page = "⚙️ Administración"
            st.rerun()

    st.divider()
    st.subheader("Personas registradas (FaceDB)")
    db_live = load_db_from_sheet()
    st.write({k: len(v) for k, v in db_live.items()})

    del_name = st.text_input("Nombre exacto a eliminar")
    if st.button("🗑️ Eliminar persona"):
        if del_name.strip() and del_name.strip() in db_live:
            delete_person_from_sheet(del_name.strip())
            st.success("Eliminado de FaceDB.")

            # Mantenerse en Administración
            st.session_state.page = "⚙️ Administración"
            st.rerun()
        else:
            st.error("No existe ese nombre.")
