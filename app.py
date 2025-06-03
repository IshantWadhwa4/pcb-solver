import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from PIL import Image
import base64
import requests
import pytesseract

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
    )
    full_prompt = expert_instruction + (prompt or "")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""
    return result

# Helper: Extract text from image using Tesseract OCR
def extract_text_from_image_with_tesseract(uploaded_image):
    try:
        image = Image.open(uploaded_image)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""

# Main logic
solve_text = None
if st.button("Solve Problem"):
    if not groq_api_key:
        st.warning("Please enter your Groq API key.")
    else:
        input_to_solve = problem_text.strip() if problem_text else ""
        if uploaded_image is not None:
            extracted_text = extract_text_from_image_with_tesseract(uploaded_image)
            if extracted_text:
                if input_to_solve:
                    input_to_solve += "\n" + extracted_text
                else:
                    input_to_solve = extracted_text
        if input_to_solve:
            solve_text = call_groq_api_stream(input_to_solve, "meta-llama/llama-prompt-guard-2-86m", groq_api_key)
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
    st.audio(audio_fp, format='audio/mp3')
    st.caption("Click play to listen to the solution.") 
