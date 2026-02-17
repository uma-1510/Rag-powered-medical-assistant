import os
from dotenv import load_dotenv
import google.generativeai as genai

def call_gemini(prompt, stream=True, model_name='gemini-1.5-pro'):
    load_dotenv()
    api_key= os.getenv("GEMINI_aPI_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    model= genai.GenerativeModel(model_name)
    if stream:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            yield chunk.text
    else:
        response = model.generate_content(prompt)
        return response.text
    

def get_truncated_gemini_answer(chunk_generator, max_words=40):
    words=[]
    for chunk in chunk_generator:
        for word in chunk.split():
            words.append(word)
            if len(words)>= max_words:
                return ' '.join(words) + '...'
            
    return ' '.join(words)

