# app.py — Asistencia facial (selfie 1 a 1) + Enrolamiento (admin)
#        + Tipo seleccionado (ENTRADA/SALIDA) SIN autodetección por horario
#        + Anti-duplicado: si duplica ENTRADA o SALIDA en el día => error fijo
#        + NO requiere ENTRADA para permitir SALIDA
#        + GPS OBLIGATORIO (no deja continuar sin permiso)
#        + CÁMARA: NO confundir "sin foto" con "sin permiso"
#        + Mensaje de puntualidad para ENTRADA (corte 07:46)
#        + Mensaje de cierre para SALIDA
#        + Google Sheets (asistencias) + FaceDB en Sheets (embeddings persistentes)
#        + Dirección (reverse geocode OSM)
#        + Versión sin OpenCV/cv2 para evitar errores en Streamlit Cloud

import base64
import io
import requests
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

from insightface.app import FaceAnalysis
from streamlit_js_eval import get_geolocation


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Asistencia Facial", page_icon="✅", layout="centered")

TZ = ZoneInfo("America/Lima")

APP_TITLE = "✅ Control de asistencia SOLVEX (Entrada/Salida)"
DEFAULT_THRESHOLD = 0.38

SHEET_NAME = st.secrets.get("SHEET_NAME", "Asistencia Facial")
WORKSHEET_NAME = st.secrets.get("WORKSHEET_NAME", "Hoja 1")
SPREADSHEET_ID = st.secrets.get("SPREADSHEET_ID", "")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")

FACE_DB_SHEET = "FaceDB"

ATT_HEADER = ["fecha_hora", "nombre", "tipo", "score", "lat", "lon", "accuracy_m", "direccion"]
FACE_HEADER = ["nombre", "emb_b64", "created_at"]

DUPLICATE_ERROR_MSG = "❌ Duplicidad de marcación, verifique marcaje."

# Corte de puntualidad
PUNCTUAL_CUTOFF = time(7, 46, 0)  # 07:46:00


# ----------------------------
# IMAGE HELPER SIN CV2
# ----------------------------
def image_file_to_bgr(uploaded_file):
    """
    Convierte la imagen tomada con st.camera_input a formato BGR,
    compatible con InsightFace, sin usar OpenCV/cv2.
    """
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = img_rgb[:, :, ::-1].copy()
    return img_bgr


# ----------------------------
# FACE MODEL (cache)
# ----------------------------
@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")

    # ctx_id=-1 fuerza CPU.
    # Es más compatible con Streamlit Cloud que ctx_id=0.
    app.prepare(ctx_id=-1, det_size=(640, 640))

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
# IDENTIFICATION
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
        return sh.add_worksheet(title=name, rows=5000, cols=20)


# ----------------------------
# SHEET HELPERS (Asistencia)
# ----------------------------
def ensure_attendance_header(ws):
    values = ws.get_all_values()

    if not values:
        ws.append_row(ATT_HEADER, value_input_option="USER_ENTERED")
        return

    ws.update("A1:H1", [ATT_HEADER])


def read_attendance_df(ws):
    values = ws.get_all_values()

    if not values:
        df = pd.DataFrame(columns=ATT_HEADER)
        df["_dt"] = pd.to_datetime([], errors="coerce")
        return df

    header = values[0]
    rows = values[1:]

    df = pd.DataFrame(rows, columns=header)

    for col in ATT_HEADER:
        if col not in df.columns:
            df[col] = ""

    df = df[ATT_HEADER].copy()

    df["_dt"] = pd.to_datetime(
        df["fecha_hora"],
        errors="coerce",
        infer_datetime_format=True,
    )

    return df


def _today_date():
    return datetime.now(TZ).date()


def marked_today(df, name, tipo):
    """
    True si ya existe ese tipo ENTRADA/SALIDA hoy para esa persona.
    """
    if df.empty:
        return False

    today = _today_date()

    sub = df[
        (df["nombre"].astype(str) == name)
        & (df["tipo"].astype(str) == tipo)
    ].copy()

    sub = sub[sub["_dt"].notna()]

    if sub.empty:
        return False

    return bool((sub["_dt"].dt.date == today).any())


# ----------------------------
# Reverse geocoding
# ----------------------------
@st.cache_data(ttl=3600)
def reverse_geocode(lat, lon):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"

        params = {
            "lat": float(lat),
            "lon": float(lon),
            "format": "json",
            "zoom": 19,
            "addressdetails": 1,
        }

        headers = {
            "User-Agent": "Asistencia-Facial-SOLVEX/1.9 (contacto@solvexing.com)"
        }

        r = requests.get(url, params=params, headers=headers, timeout=4)

        if r.status_code != 200:
            return ""

        data = r.json()
        addr = data.get("address", {})

        road = (
            addr.get("road")
            or addr.get("pedestrian")
            or addr.get("footway")
            or ""
        )

        house = addr.get("house_number") or ""

        neighbourhood = (
            addr.get("neighbourhood")
            or addr.get("suburb")
            or addr.get("quarter")
            or ""
        )

        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or ""
        )

        state = addr.get("state") or ""
        country = addr.get("country") or ""

        parts = []

        if road:
            parts.append(road + (f" {house}" if house else ""))
        elif house:
            parts.append(house)

        for p in [neighbourhood, city, state, country]:
            if p and p not in parts:
                parts.append(p)

        return ", ".join([p for p in parts if p]).strip()

    except Exception:
        return ""


def append_attendance(ws, name, tipo, score, geo):
    now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

    lat, lon, acc, direccion = "", "", "", ""

    lat = geo["coords"].get("latitude", "")
    lon = geo["coords"].get("longitude", "")
    acc = geo["coords"].get("accuracy", "")

    try:
        if lat != "" and lon != "":
            direccion = reverse_geocode(lat, lon)
    except Exception:
        direccion = ""

    ws.append_row(
        [now, name, tipo, round(float(score), 4), lat, lon, acc, direccion],
        value_input_option="USER_ENTERED",
    )


# ----------------------------
# FACE DB in Google Sheets
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

    rows = [[name, emb_to_b64(emb), now] for emb in embs]

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
# UI
# ----------------------------
st.title(APP_TITLE)
st.caption(
    "Selecciona ENTRADA o SALIDA, permite GPS y cámara, toma una selfie "
    "y se registrará en Google Sheets."
)

if "page" not in st.session_state:
    st.session_state.page = "📸 Marcar asistencia"

page = st.sidebar.radio(
    "Menú",
    ["📸 Marcar asistencia", "⚙️ Administración"],
    index=0 if st.session_state.page == "📸 Marcar asistencia" else 1,
)

st.session_state.page = page

db = load_db_from_sheet()


# ----------------------------
# PÁGINA 1: MARCAR ASISTENCIA
# ----------------------------
if page == "📸 Marcar asistencia":
    threshold = st.slider(
        "Umbral de reconocimiento",
        0.30,
        0.60,
        float(DEFAULT_THRESHOLD),
        0.01,
    )

    st.write(f"Personas registradas: **{len(db)}**")

    now_dt = datetime.now(TZ)

    st.info(f"Hora actual: **{now_dt.strftime('%Y-%m-%d %H:%M:%S')}**")

    tipo = st.radio(
        "¿Qué deseas registrar?",
        ["ENTRADA", "SALIDA"],
        horizontal=True,
        index=0,
    )

    # GPS obligatorio
    st.subheader("📍 Geolocalización (OBLIGATORIA)")
    st.caption(
        "Si no te aparece el permiso, revisa el ícono 🔒/ubicación del navegador "
        "y habilítalo."
    )

    geo = get_geolocation()

    if not geo or not geo.get("coords"):
        st.error("⚠️ Debes habilitar el permiso de ubicación (GPS) para registrar asistencia.")
        st.info("👉 Chrome/Edge: clic en el ícono 🔒 junto a la URL → Ubicación → Permitir → recargar.")
        st.stop()

    st.success(
        f"GPS OK: lat={geo['coords'].get('latitude')}, "
        f"lon={geo['coords'].get('longitude')}, "
        f"±{geo['coords'].get('accuracy')} m"
    )

    # Cámara obligatoria
    st.subheader("📸 Cámara (OBLIGATORIA)")
    st.caption(
        "Cuando estés listo, presiona **Take Photo**. "
        "Si no ves la imagen, habilita el permiso de cámara en el navegador."
    )

    img_file = st.camera_input("Toma tu selfie")

    if img_file is None:
        st.warning("📸 Toma la foto para continuar, usando el botón **Take Photo**.")
        st.stop()

    # Procesar selfie sin cv2
    img_bgr = image_file_to_bgr(img_file)

    emb = get_face_embedding(img_bgr)

    if emb is None:
        st.error("No se detectó rostro. Intenta con mejor luz y más cerca.")
        st.stop()

    if len(db) == 0:
        st.error("❌ No hay personas registradas. Ve a ⚙️ Administración para enrolar.")
        st.stop()

    try:
        name, score = identify(db, emb, threshold)

        if name is None:
            st.error(f"No identificado. score={score:.3f} | umbral={threshold:.2f}")
            st.stop()

        ws = open_worksheet(WORKSHEET_NAME)
        ensure_attendance_header(ws)
        df = read_attendance_df(ws)

        if marked_today(df, name, tipo):
            st.error(DUPLICATE_ERROR_MSG)
            st.stop()

        append_attendance(ws, name, tipo, score, geo)

        if tipo == "ENTRADA":
            if now_dt.time() <= PUNCTUAL_CUTOFF:
                st.success("Bienvenido a Solvex, agradecemos tu puntualidad! ✅⏱️")
            else:
                st.warning("Bienvenido a Solvex! Estás llegando un poco tarde ⏰😅")
        else:
            st.success("¡Gran trabajo el de este día, te esperamos mañana! 💪🌟")

        st.success(f"✅ {tipo} registrada: {name} | score={score:.3f}")

        st.subheader("Últimos registros")

        st.dataframe(
            read_attendance_df(ws).tail(15).drop(columns=["_dt"], errors="ignore"),
            use_container_width=True,
        )

    except Exception as e:
        st.error("Error conectando con Google Sheets. Detalle:")
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

    st.write("Recomendado: **3 selfies**: frente, leve izquierda y leve derecha.")

    new_name = st.text_input("Nombre exacto")

    n_photos = st.selectbox(
        "Cantidad de selfies",
        [1, 2, 3, 4, 5],
        index=2,
    )

    if "enroll_embs" not in st.session_state:
        st.session_state.enroll_embs = []

    enroll_img = st.camera_input("Selfie para enrolamiento", key="enroll_cam")

    if enroll_img is not None:
        bgr = image_file_to_bgr(enroll_img)
        emb = get_face_embedding(bgr)

        if emb is None:
            st.error("No se detectó rostro en esta selfie.")
        else:
            st.session_state.enroll_embs.append(emb)
            st.success(
                f"Selfie guardada "
                f"({len(st.session_state.enroll_embs)}/{int(n_photos)})"
            )

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

            st.session_state.page = "⚙️ Administración"

            st.rerun()

        else:
            st.error("No existe ese nombre.")