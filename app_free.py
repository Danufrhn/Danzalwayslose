"""
TimeCapsule - TRULY FREE Version (Streamlit)
------------------------------------------------
Update penting (Juli 2026): Google Gemini API SUDAH MENUTUP akses gratis
untuk generate gambar lewat API (limit free tier = 0 untuk model image).
Jadi versi ini tidak memakai API generate-gambar berbayar apa pun.

Teknologi yang dipakai (SEMUA gratis, tanpa API key untuk gambar/video):
- Google Gemini API (teks saja, model gemini-3.5-flash-lite) -> generate cerita/deskripsi karakter (free tier)
- PIL + OpenCV (lokal)                                        -> filter visual foto sesuai era (sepia, grain, vignette, dll)
- OpenCV (lokal)                                               -> "video" dari efek Ken Burns (zoom/pan) di atas foto yang sudah difilter

Kalau GOOGLE_API_KEY tidak diisi sama sekali, app tetap bisa jalan
(cerita/deskripsi pakai teks template, tanpa AI) - jadi bisa 100% offline juga.

Cara dapat GOOGLE_API_KEY (gratis, tanpa kartu kredit, untuk teks saja):
1. Buka https://aistudio.google.com/apikey
2. Login dengan akun Google, klik "Create API key"
3. JANGAN pernah share/screenshot key ini ke siapa pun

Jalankan:
    streamlit run app_truly_free.py

Install dependencies:
    pip install streamlit pillow numpy opencv-python-headless google-genai python-dotenv
"""

import os
import uuid
from datetime import datetime
from enum import Enum

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

os.makedirs("intermediate_images", exist_ok=True)
os.makedirs("final_videos", exist_ok=True)


# ----------------------------------------------------------------------------
# Pilihan-pilihan
# ----------------------------------------------------------------------------
class TimePeriodOptions(Enum):
    ANCIENT = "Ancient Times (Before 500 AD)"
    MEDIEVAL = "Medieval (500-1500 AD)"
    VICTORIAN = "Victorian Era (1800-1900)"
    EARLY_20TH = "Early 20th Century (1900-1950)"
    MODERN = "Modern Era (1990-Present)"
    FUTURISTIC = "Futuristic (Near Future)"


class ProfessionOptions(Enum):
    WARRIOR = "Warrior/Soldier"
    SCHOLAR = "Scholar/Teacher"
    ARTISAN = "Artisan/Craftsperson"
    EXPLORER = "Explorer/Adventurer"
    NOBLE = "Noble/Aristocrat"


class AnimationOptions(Enum):
    ZOOM_IN = "Zoom In (perlahan mendekat)"
    ZOOM_OUT = "Zoom Out (perlahan menjauh)"
    PAN_LEFT_RIGHT = "Pan Kiri ke Kanan"


# Mapping efek filter per era (semua diproses lokal, tanpa API)
FILTER_BY_ERA = {
    TimePeriodOptions.ANCIENT: "sepia_heavy",
    TimePeriodOptions.MEDIEVAL: "sepia_heavy",
    TimePeriodOptions.VICTORIAN: "bw_grain",
    TimePeriodOptions.EARLY_20TH: "bw_grain",
    TimePeriodOptions.MODERN: "none",
    TimePeriodOptions.FUTURISTIC: "cyber_glow",
}


# ----------------------------------------------------------------------------
# Fungsi teks (Gemini, gratis) - opsional
# ----------------------------------------------------------------------------
def make_story_text(time_period, profession) -> str:
    """Buat narasi singkat dengan Gemini (gratis, teks saja). Fallback ke template kalau gagal/tidak ada key."""
    fallback = (
        f"Kamu melintasi waktu menuju {time_period.value}, menjalani hidup sebagai "
        f"seorang {profession.value.lower()}. Sebuah perjalanan yang penuh makna."
    )
    if not GOOGLE_API_KEY:
        return fallback
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        prompt = (
            f"Tulis narasi singkat (maksimal 3 kalimat, bahasa Indonesia) tentang seseorang "
            f"yang berpindah waktu menjadi seorang {profession.value} pada masa "
            f"{time_period.value}. Buat terasa personal dan sedikit dramatis, aman untuk segala usia."
        )
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        st.info(f"(Gemini tidak tersedia, pakai teks template. Detail: {e})")
        return fallback


# ----------------------------------------------------------------------------
# Fungsi filter foto (lokal, 100% gratis, tanpa API)
# ----------------------------------------------------------------------------
def apply_sepia(img_rgb: np.ndarray) -> np.ndarray:
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ])
    sepia_img = img_rgb @ sepia_filter.T
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


def apply_vignette(img_rgb: np.ndarray, strength=0.8) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w * 0.5)
    kernel_y = cv2.getGaussianKernel(h, h * 0.5)
    kernel = kernel_y @ kernel_x.T
    mask = kernel / kernel.max()
    mask = mask * strength + (1 - strength)
    vignette = img_rgb.astype(np.float64) * mask[:, :, np.newaxis]
    return np.clip(vignette, 0, 255).astype(np.uint8)


def apply_grain(img_rgb: np.ndarray, amount=18) -> np.ndarray:
    noise = np.random.normal(0, amount, img_rgb.shape)
    noisy = img_rgb.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_era_filter(image_path: str, filter_name: str) -> str:
    """Terapkan filter visual sesuai era, simpan hasilnya, return path baru."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    if filter_name == "sepia_heavy":
        arr = apply_sepia(arr)
        arr = apply_vignette(arr, strength=0.6)
        arr = apply_grain(arr, amount=10)
    elif filter_name == "bw_grain":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        arr = apply_grain(arr, amount=20)
        arr = apply_vignette(arr, strength=0.7)
    elif filter_name == "cyber_glow":
        img_pil = Image.fromarray(arr)
        img_pil = ImageEnhance.Color(img_pil).enhance(1.6)
        img_pil = ImageEnhance.Contrast(img_pil).enhance(1.25)
        img_pil = ImageEnhance.Brightness(img_pil).enhance(1.05)
        arr = np.array(img_pil)
        # tint kebiruan ala futuristik
        tint = np.array([0.85, 1.0, 1.25])
        arr = np.clip(arr.astype(np.float64) * tint, 0, 255).astype(np.uint8)
    # filter_name == "none" -> tidak diapa-apakan

    result_img = Image.fromarray(arr)
    filename = f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.png"
    filepath = os.path.join("intermediate_images", filename)
    result_img.save(filepath)
    return filepath


def create_video_from_image(image_path, animation: "AnimationOptions", duration_sec=5, fps=24):
    """Buat 'video' dari satu gambar dengan efek Ken Burns (zoom/pan), 100% lokal, tanpa API."""
    img = Image.open(image_path).convert("RGB")
    out_w, out_h = 1280, 720

    scale_factor = 1.3
    big_w, big_h = int(out_w * scale_factor), int(out_h * scale_factor)
    img_resized = img.resize((big_w, big_h))
    frame_big = np.array(img_resized)

    total_frames = int(duration_sec * fps)
    filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.mp4"
    filepath = os.path.join("final_videos", filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (out_w, out_h))

    max_x_offset = big_w - out_w
    max_y_offset = big_h - out_h

    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)

        if animation == AnimationOptions.ZOOM_IN:
            x_off = int((max_x_offset / 2) * t)
            y_off = int((max_y_offset / 2) * t)
        elif animation == AnimationOptions.ZOOM_OUT:
            x_off = int((max_x_offset / 2) * (1 - t))
            y_off = int((max_y_offset / 2) * (1 - t))
        else:
            x_off = int(max_x_offset * t)
            y_off = max_y_offset // 2

        crop = frame_big[y_off:y_off + out_h, x_off:x_off + out_w]
        frame_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return filepath


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="TimeCapsule (100% Gratis)", page_icon="🕰️")
st.title("🕰️ TimeCapsule - 100% Gratis")
st.caption("Tanpa API berbayar sama sekali. Filter foto & video dibuat lokal.")

with st.sidebar:
    st.subheader("Status")
    if GOOGLE_API_KEY:
        st.write("✅ GOOGLE_API_KEY (untuk narasi teks AI)")
    else:
        st.write("⚪ GOOGLE_API_KEY tidak diisi — narasi pakai teks template (tetap jalan normal)")
    st.caption("Filter foto & video 100% diproses lokal, tidak butuh API sama sekali.")

uploaded_photo = st.file_uploader("Upload foto", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    profession_label = st.selectbox("Profesi", [p.value for p in ProfessionOptions])
    animation_label = st.selectbox("Efek Video", [a.value for a in AnimationOptions])
with col2:
    time_period_label = st.selectbox("Periode Waktu", [t.value for t in TimePeriodOptions])

time_period = next(t for t in TimePeriodOptions if t.value == time_period_label)
profession = next(p for p in ProfessionOptions if p.value == profession_label)
animation = next(a for a in AnimationOptions if a.value == animation_label)

if st.button("🚀 Generate", type="primary", use_container_width=True):
    if not uploaded_photo:
        st.error("Upload foto dulu ya.")
        st.stop()

    photo_path = os.path.join("intermediate_images", f"input_{uuid.uuid4()}.jpg")
    Image.open(uploaded_photo).convert("RGB").save(photo_path)

    with st.status("Membuat TimeCapsule kamu...", expanded=True) as status:
        st.write("Membuat narasi cerita...")
        story = make_story_text(time_period, profession)
        st.write(f"→ {story}")

        st.write("Menerapkan filter visual sesuai era (lokal)...")
        filter_name = FILTER_BY_ERA[time_period]
        filtered_path = apply_era_filter(photo_path, filter_name)

        st.write("Membuat animasi video dari foto (lokal)...")
        video_path = create_video_from_image(filtered_path, animation)

        status.update(label="Selesai! 🎉", state="complete")

    st.subheader("Hasil")
    st.info(story)
    c1, c2 = st.columns(2)
    with c1:
        st.image(filtered_path, caption="Foto dengan filter era")
    with c2:
        st.video(video_path)
