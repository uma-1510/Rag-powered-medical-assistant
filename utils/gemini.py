import os
from dotenv import load_dotenv
import google.generativeai as genai

# INITIALIZE ONCE
DEFAULT_MODEL = "gemini-flash-latest" 
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set")

genai.configure(api_key=API_KEY)

  # MUCH cheaper than pro
for m in genai.list_models():
    print(m.name)

def call_gemini(prompt, stream=False, model_name=DEFAULT_MODEL):

    model = genai.GenerativeModel(model_name)
    
    # Not Stream Mode
    if not stream:
        response = model.generate_content(prompt)
        return response.text

    # ---------- STREAM MODE ----------
    response = model.generate_content(prompt, stream=True)

    def stream_generator():
        for chunk in response:
            if chunk.text:
                yield chunk.text

    return stream_generator()


# OPTIONAL preview helper (cheap debugging)
def get_truncated_gemini_answer(chunk_generator, max_words=40):

    words = []

    for chunk in chunk_generator:
        for word in chunk.split():
            words.append(word)
            if len(words) >= max_words:
                return " ".join(words) + "..."

    return " ".join(words)
