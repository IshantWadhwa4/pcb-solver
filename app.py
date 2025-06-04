import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from PIL import Image
import requests
import base64

st.title("PCM Problem Solver with Groq API and gTTS")

# 1. Subject selection
task = st.selectbox("Solve problem for:", ["Physics", "Chemistry", "Maths"])

# 2. Groq API key input
groq_api_key = st.text_input("Enter your Groq API key:", type="password")

# 3. Problem input (text or image)
problem_text = st.text_area("Enter your problem (optional):")
uploaded_image = st.file_uploader("Upload an image or take a picture (optional)", type=["jpg", "jpeg", "png"])

# Helper: Call Groq API using official client (streaming)
def call_groq_api_stream(prompt, model, api_key):
    client = Groq(api_key=api_key)
    expert_instruction = (
        f"You are a {task} expert.\n"
        "You will be given a question and you need to solve it step by step.\n"
        "Answer in a mix of English and Hindi as if you are an Indian teacher explaining to a student.\n"
        "Limit your response to 500 tokens only follow the instructions strictly."
    )
    full_prompt = expert_instruction + (prompt or "")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        stream=True,
        stop=None,
    )
    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""
    return result

# Helper: OCR.Space API for OCR
def ocr_space_file_upload(image_file, api_key='K88884750088957'):
    payload = {
        'isOverlayRequired': False,
        'apikey': api_key,
        'language': 'eng',
    }
    files = {'file': image_file}
    r = requests.post('https://api.ocr.space/parse/image',
                      files=files,
                      data=payload,
                      )
    result = r.json()
    try:
        return result['ParsedResults'][0]['ParsedText']
    except Exception:
        return ""

def get_audio_download_link(audio_fp, filename="solution.mp3"):
    audio_fp.seek(0)
    b64 = base64.b64encode(audio_fp.read()).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download audio</a>'
    return href

# Main logic
solve_text = None
if st.button("Solve Problem"):
    if not groq_api_key:
        st.warning("Please enter your Groq API key.")
    else:
        input_to_solve = problem_text.strip() if problem_text else ""
        if uploaded_image is not None:
            extracted_text = ocr_space_file_upload(uploaded_image)
            st.write(extracted_text)
            if extracted_text:
                if input_to_solve:
                    input_to_solve += "\n" + extracted_text
                else:
                    input_to_solve = extracted_text
        if input_to_solve:
            solve_text = call_groq_api_stream(input_to_solve, "llama3-8b-8192", groq_api_key)
        else:
            st.warning("Please provide a problem in text or upload an image.")

if solve_text:
    st.subheader("Solution:")
    st.write(solve_text)
    # gTTS for audio output
    tts = gTTS(solve_text)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    st.markdown(get_audio_download_link(audio_fp), unsafe_allow_html=True)
    st.caption("If audio does not play, use the download link above (especially on iOS).") 
    st.audio(audio_fp, format='audio/mp3')
    
