"""
TimeCapsule - Simple Version (Streamlit)
------------------------------------------
Alur sederhana:
1. Upload foto
2. Pilih Etnis, Periode Waktu, Profesi, Aksi
3. Generate gambar AI (RunwayML)
4. Generate video dari gambar (fal.ai Seedance)
5. Tampilkan & download hasil

Jalankan:
    streamlit run app.py

Isi API key di file .env (lokal) atau Secrets (Streamlit Cloud):
    OPENAI_API_KEY
    RUNWAYML_API_SECRET
    FAL_KEY

Install dependencies:
    pip install streamlit pillow requests openai runwayml fal-client python-dotenv
"""

import os
import re
import uuid
from datetime import datetime
from enum import Enum

import requests
import base64
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
RUNWAYML_API_SECRET = get_secret("RUNWAYML_API_SECRET")
FAL_KEY = get_secret("FAL_KEY")

if RUNWAYML_API_SECRET:
    os.environ["RUNWAYML_API_SECRET"] = RUNWAYML_API_SECRET
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

os.makedirs("intermediate_images", exist_ok=True)
os.makedirs("final_videos", exist_ok=True)


# ----------------------------------------------------------------------------
# Pilihan-pilihan
# ----------------------------------------------------------------------------
class EthnicityOptions(Enum):
    CAUCASIAN = "Caucasian"
    AFRICAN = "African"
    ASIAN = "Asian"
    HISPANIC = "Hispanic"
    MIDDLE_EASTERN = "Middle Eastern"
    MIXED = "Mixed Heritage"


class TimePeriodOptions(Enum):
    ANCIENT = "Ancient Times (Before 500 AD)"
    MEDIEVAL = "Medieval (500-1500 AD)"
    RENAISSANCE = "Renaissance (1400-1600)"
    COLONIAL = "Colonial Era (1600-1800)"
    VICTORIAN = "Victorian Era (1800-1900)"
    MODERN = "Modern Era (1990-Present)"
    FUTURISTIC = "Futuristic (Near Future)"


class ProfessionOptions(Enum):
    WARRIOR = "Warrior/Soldier"
    SCHOLAR = "Scholar/Teacher"
    ARTISAN = "Artisan/Craftsperson"
    EXPLORER = "Explorer/Adventurer"
    NOBLE = "Noble/Aristocrat"


class ActionOptions(Enum):
    SELFIE = "Taking a selfie from camera view"
    WALKING = "Simple walking"
    WORK_ACTION = "Performing work/professional action"
    CELEBRATION = "Celebrating/cheering"


# ----------------------------------------------------------------------------
# Fungsi inti
# ----------------------------------------------------------------------------
def image_to_data_uri(filepath: str) -> str:
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def sanitize_prompt(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt.strip())
    prompt = re.sub(r"[^\w\s,.-]", "", prompt)
    return prompt or "Professional portrait photograph"


def make_image_prompt(ethnicity, time_period, profession, action) -> str:
    """Buat prompt gambar dengan OpenAI, fallback ke prompt sederhana kalau gagal."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        base_prompt = (
            f"Create a simple, clean prompt for AI image generation: "
            f"Ethnicity: {ethnicity.value}, Profession: {profession.value}, "
            f"Time period: {time_period.value}, Action: {action.value}. "
            f"Show appropriate clothing and setting, unique background, "
            f"appropriate for all ages. Maximum 30 words."
        )
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": base_prompt}],
            max_tokens=80,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return (
            f"{ethnicity.value} {profession.value} from {time_period.value} "
            f"performing {action.value}, professional portrait"
        )


def make_video_prompt(ethnicity, time_period, profession, action, image_prompt) -> str:
    """Buat prompt video dengan OpenAI, fallback ke prompt sederhana kalau gagal."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        base_prompt = (
            f"Context: {image_prompt}. Generate a short, safe, family-friendly video "
            f"prompt for a {ethnicity.value} {profession.value} from {time_period.value} "
            f"{action.value}. Focus entirely on the action, keep it realistic, "
            f"avoid violence or controversial topics."
        )
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": base_prompt}],
            max_tokens=60,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip() + " Keep face of the @person consistent."
    except Exception:
        return f"{ethnicity.value} {profession.value} from {time_period.value} {action.value}"


def generate_image(photo_path, prompt):
    """Generate gambar baru dari foto + prompt menggunakan RunwayML."""
    from runwayml import RunwayML
    client = RunwayML()

    task = client.text_to_image.create(
        model="gen4_image",
        prompt_text=sanitize_prompt(prompt),
        ratio="1360:768",
        reference_images=[{"uri": image_to_data_uri(photo_path)}],
    ).wait_for_task_output()

    image_url = task.output[0]
    resp = requests.get(image_url)
    resp.raise_for_status()

    filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.png"
    filepath = os.path.join("intermediate_images", filename)
    with open(filepath, "wb") as f:
        f.write(resp.content)
    return filepath


def generate_video(image_path, prompt):
    """Generate video dari gambar menggunakan fal.ai Seedance V1 Pro."""
    import fal_client

    image_url = fal_client.upload_file(image_path)
    result = fal_client.subscribe(
        "fal-ai/bytedance/seedance/v1/pro/image-to-video",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "resolution": "720p",
            "duration": 10,
        },
        with_logs=False,
    )

    if not result or "video" not in result or "url" not in result["video"]:
        raise RuntimeError("Tidak ada URL video pada hasil fal.ai")

    resp = requests.get(result["video"]["url"])
    resp.raise_for_status()

    filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.mp4"
    filepath = os.path.join("final_videos", filename)
    with open(filepath, "wb") as f:
        f.write(resp.content)
    return filepath


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="TimeCapsule (Simple)", page_icon="🕰️")
st.title("🕰️ TimeCapsule - Simple")
st.caption("Upload foto, pilih karakter, dapatkan gambar & video AI.")

with st.sidebar:
    st.subheader("Status API Key")
    for name, val in [
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("RUNWAYML_API_SECRET", RUNWAYML_API_SECRET),
        ("FAL_KEY", FAL_KEY),
    ]:
        st.write(("✅ " if val else "❌ ") + name)

uploaded_photo = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    ethnicity_label = st.selectbox("Etnis", [e.value for e in EthnicityOptions])
    profession_label = st.selectbox("Profesi", [p.value for p in ProfessionOptions])
with col2:
    time_period_label = st.selectbox("Periode Waktu", [t.value for t in TimePeriodOptions])
    action_label = st.selectbox("Aksi", [a.value for a in ActionOptions])

ethnicity = next(e for e in EthnicityOptions if e.value == ethnicity_label)
time_period = next(t for t in TimePeriodOptions if t.value == time_period_label)
profession = next(p for p in ProfessionOptions if p.value == profession_label)
action = next(a for a in ActionOptions if a.value == action_label)

if st.button("🚀 Generate", type="primary", use_container_width=True):
    if not uploaded_photo:
        st.error("Upload foto dulu ya.")
        st.stop()
    if not (RUNWAYML_API_SECRET and FAL_KEY):
        st.error("RUNWAYML_API_SECRET dan/atau FAL_KEY belum diisi.")
        st.stop()

    photo_path = os.path.join("intermediate_images", f"input_{uuid.uuid4()}.jpg")
    Image.open(uploaded_photo).convert("RGB").save(photo_path)

    with st.status("Membuat TimeCapsule kamu...", expanded=True) as status:
        st.write("Membuat prompt gambar...")
        image_prompt = make_image_prompt(ethnicity, time_period, profession, action)

        st.write("Generate gambar dengan RunwayML...")
        try:
            image_path = generate_image(photo_path, image_prompt)
        except Exception as e:
            status.update(label="Gagal generate gambar", state="error")
            st.error(f"Error: {e}")
            st.stop()

        st.write("Membuat prompt video...")
        video_prompt = make_video_prompt(ethnicity, time_period, profession, action, image_prompt)

        st.write("Generate video dengan fal.ai (bisa 1-3 menit)...")
        try:
            video_path = generate_video(image_path, video_prompt)
        except Exception as e:
            status.update(label="Gagal generate video", state="error")
            st.error(f"Error: {e}")
            st.stop()

        status.update(label="Selesai! 🎉", state="complete")

    st.subheader("Hasil")
    c1, c2 = st.columns(2)
    with c1:
        st.image(image_path, caption="Gambar AI")
    with c2:
        st.video(video_path)
