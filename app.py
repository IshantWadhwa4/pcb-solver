import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from PIL import Image
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

# Helper: Extract text from image using Groq API OCR model
def extract_text_from_image(image_bytes, api_key):
    client = Groq(api_key=api_key)
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    ocr_instruction = (
        "You are an OCR expert.\n"
        "You will be given an image and you need to extract the text from it. "
        "Text may contain maths equations, so you need to give exact text we use in maths equations.\n"
        f"You are a {task} expert.\n"
        "You will be given a question and you need to solve it step by step.\n"
        "Answer in a mix of English and Hindi as if you are an Indian teacher explaining to a student.\n"
    )
    prompt = f"Extract the text from this image (base64-encoded): "
    full_prompt = ocr_instruction + prompt
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[{"role": "user", "content": full_prompt,
                   "type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}],
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

# Main logic
solve_text = None
if st.button("Solve Problem"):
    if not groq_api_key:
        st.warning("Please enter your Groq API key.")
    else:
        input_to_solve = problem_text.strip() if problem_text else ""
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            extracted_text = extract_text_from_image(image_bytes, groq_api_key)
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
    st.audio(audio_fp, format='audio/mp3')
    st.caption("Click play to listen to the solution.") 
