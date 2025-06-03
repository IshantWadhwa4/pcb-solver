# pcb-solver
Solve PCB problems for upto 12th class students
# PCM Problem Solver

This is a Streamlit web app that allows users to solve Physics, Chemistry, or Maths problems using the Groq API. Users can input problems as text or upload/capture an image. The app uses gTTS to provide audio playback of the solution.

## Features
- Select subject: Physics, Chemistry, or Maths
- Enter your Groq API key
- Input a problem as text or upload/capture an image
- If image is uploaded, text is extracted using Groq API
- Solution is generated using Groq API
- Listen to the solution using Google Text-to-Speech (gTTS)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run pcm_solver_app.py
   ```

3. **Usage:**
   - Enter your Groq API key (get it from https://console.groq.com/)
   - Select the subject
   - Enter the problem as text or upload/capture an image
   - Click "Solve Problem" to get the solution and listen to it

## Notes
- The app uses the following Groq models:
  - `meta-llama/llama-prompt-guard-2-86m` for solving problems
  - `meta-llama/llama-4-maverick-17b-128e-instruct` for extracting text from images
- Make sure your Groq API key has access to these models. 
