import base64
import requests
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
WORKSHEET_NAME = st.secrets.get("WORKSHEET_NAME", "Hoja 1")
SPREADSHEET_ID = st.secrets.get("SPREADSHEET_ID", "")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")

FACE_DB_SHEET = "FaceDB"

ENTRY_TARGET = time(7, 45)
EXIT_TARGET = time(17, 15)

ENTRY_WINDOW_START = time(7, 0)
ENTRY_WINDOW_END = time(11, 59)
EXIT_WINDOW_START = time(12, 0)
EXIT_WINDOW_END = time(20, 30)

ATT_HEADER = [
    "fecha_hora",
    "nombre",
    "tipo",
    "score",
    "lat",
    "lon",
    "accuracy_m",
    "direccion",
]

FACE_HEADER = ["nombre", "emb_b64", "created_at"]

# ----------------------------
# FACE MODEL
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
# SIMILARITY
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
# GOOGLE SHEETS
# ----------------------------
@st.cache_resource
def get_gs_client():
    creds_info = st.secrets.get("gcp_service_account")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(dict(creds_info), scopes=scopes)
    return gspread.authorize(creds)

def open_spreadsheet():
    gc = get_gs_client()
    return gc.open_by_key(SPREADSHEET_ID) if SPREADSHEET_ID else gc.open(SHEET_NAME)

def open_worksheet(name):
    sh = open_spreadsheet()
    try:
        return sh.worksheet(name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=name, rows=2000, cols=20)

# ----------------------------
# HEADERS
# ----------------------------
def ensure_attendance_header(ws):
    values = ws.get_all_values()
    if not values:
        ws.append_row(ATT_HEADER)
    else:
        ws.update("A1:H1", [ATT_HEADER])

def ensure_face_header(ws):
    values = ws.get_all_values()
    if not values:
        ws.append_row(FACE_HEADER)
    else:
        ws.update("A1:C1", [FACE_HEADER])

# ----------------------------
# FACE DB (SHEETS)
# ----------------------------
def emb_to_b64(emb):
    return base64.b64encode(emb.astype(np.float32).tobytes()).decode()

def b64_to_emb(s):
    try:
        emb = np.frombuffer(base64.b64decode(s), dtype=np.float32)
        return emb if emb.shape[0] == 512 else None
    except Exception:
        return None

@st.cache_data(ttl=30)
def load_db():
    ws = open_worksheet(FACE_DB_SHEET)
    ensure_face_header(ws)
    rows = ws.get_all_values()[1:]
    db = {}
    for r in rows:
        emb = b64_to_emb(r[1])
        if emb is not None:
            db.setdefault(r[0], []).append(emb)
    return db

def add_embeddings(name, embs):
    ws = open_worksheet(FACE_DB_SHEET)
    ensure_face_header(ws)
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    rows = [[name, emb_to_b64(e), now] for e in embs]
    ws.append_rows(rows)
    load_db.clear()

# ----------------------------
# GEOLOCALIZACIÓN → DIRECCIÓN
# ----------------------------
@st.cache_data(ttl=3600)
def reverse_geocode(lat, lon):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "zoom": 18,
            },
            headers={"User-Agent": "Asistencia-Facial-SOLVEX"},
            timeout=5,
        )
        return r.json().get("display_name", "") if r.status_code == 200 else ""
    except Exception:
        return ""

def append_attendance(ws, name, tipo, score, geo):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
    lat = lon = acc = direccion = ""

    if geo and geo.get("coords"):
        lat = geo["coords"].get("latitude")
        lon = geo["coords"].get("longitude")
        acc = geo["coords"].get("accuracy")
        if lat and lon:
            direccion = reverse_geocode(lat, lon)

    ws.append_row([now, name, tipo, round(score, 4), lat, lon, acc, direccion])

# ----------------------------
# LOGIC
# ----------------------------
def decide_tipo(t):
    if ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END:
        return "ENTRADA"
    if EXIT_WINDOW_START <= t <= EXIT_WINDOW_END:
        return "SALIDA"
    return None

# ----------------------------
# UI
# ----------------------------
st.title(APP_TITLE)

if "page" not in st.session_state:
    st.session_state.page = "📸 Marcar asistencia"

page = st.sidebar.radio("Menú", ["📸 Marcar asistencia", "⚙️ Administración"])
st.session_state.page = page

db = load_db()

# ---------- MARCAR ----------
if page == "📸 Marcar asistencia":
    geo = get_geolocation()
    img = st.camera_input("Toma tu selfie")

    if img:
        emb = get_face_embedding(cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), 1))
        name, score = identify(db, emb, DEFAULT_THRESHOLD)
        ws = open_worksheet(WORKSHEET_NAME)
        ensure_attendance_header(ws)
        tipo = decide_tipo(datetime.now(TZ).time())
        append_attendance(ws, name, tipo, score, geo)
        st.success(f"{tipo} registrada: {name}")

# ---------- ADMIN ----------
if page == "⚙️ Administración":
    pwd = st.text_input("Contraseña admin", type="password")
    if ADMIN_PASSWORD and pwd != ADMIN_PASSWORD:
        st.stop()

    name = st.text_input("Nombre")
    if "embs" not in st.session_state:
        st.session_state.embs = []

    img = st.camera_input("Selfie")
    if img:
        emb = get_face_embedding(cv2.imdecode(np.frombuffer(img.getvalue(), np.uint8), 1))
        if emb is not None:
            st.session_state.embs.append(emb)

    if st.button("Guardar"):
        add_embeddings(name, st.session_state.embs)
        st.session_state.embs = []
        st.success("Registrado")
        st.rerun()
