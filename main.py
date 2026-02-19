"""
MAIN APPLICATION ENTRYPOINT

This file orchestrates the full Medical RAG pipeline:

User Question
    ↓
Local Retrieval (FAISS)
    ↓
Confidence Evaluation
    ├── High confidence → Direct MedQuAD answer (NO LLM call)
    └── Medium confidence → Gemini synthesis
    ↓
Emergency Detection
    ↓
Grounded Response + Citations

Goal:
Reduce Gemini usage (cost saving)
Prevent hallucinations
Provide explainable medical answers
"""

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# --- Internal modules ---
from utils.retriever import retrieve, compute_confidence, has_direct_answer
from utils.prompt_builder import build_prompt
from utils.safety import detect_emergency
from utils.gemini import call_gemini


# Load environment variables (.env)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)


# HOME ROUTE

@app.route('/')
def home():
    """
    Serves frontend UI.
    """
    return render_template('index.html')


# MAIN QUESTION ENDPOINT

@app.route('/ask', methods=['POST'])
def ask():
    """
    Core medical QA pipeline.

    Steps:
    1. Receive user query
    2. Retrieve relevant medical chunks locally
    3. Compute retrieval confidence
    4. Skip Gemini if answer already exists
    5. Otherwise synthesize using Gemini
    6. Attach sources + safety warnings
    """

    # STEP 0: READ INPUT
    data = request.get_json() or {}
    query = data.get('question', '').strip()

    if not query:
        return jsonify({"error": "Empty question"}), 400

    try:
        # STEP 1 — LOCAL RETRIEVAL (NO LLM COST)
        
        results = retrieve(query, k=5)

        if not results:
            return jsonify({
                "answer": (
                    "I could not find reliable medical information "
                    "for this question. Please consult a healthcare professional."
                ),
                "sources": []
            })

        # STEP 2 — CONFIDENCE SCORING
        # Determines how strongly query matches dataset
        confidence = compute_confidence(results)
        print(f"Retrieval confidence: {confidence:.3f}")

        # STEP 3 — EMERGENCY DETECTION (LOCAL RULES)
        # Runs BEFORE LLM for safety
        emergency_flag = detect_emergency(query, results)

        # STEP 4 — LLM SKIP LOGIC (MAJOR COST SAVER)
        # If MedQuAD already contains strong answer,
        # we trust dataset instead of calling Gemini.
        if has_direct_answer(results):

            print("✅ High confidence match — skipping Gemini")

            # Direct grounded answer from dataset
            answer = results[0]["answer"]

        else:
            # Gemini only used when synthesis is required
            print("🤖 Using Gemini synthesis")

            gemini_prompt = build_prompt(
                query,
                results,
                emergency_flag=emergency_flag
            )

            # Single LLM call (optimized)
            answer = call_gemini(gemini_prompt, stream=False)

        # STEP 5 — ADD SAFETY WARNING IF NEEDED
        if emergency_flag:
            answer += (
                "\n\n⚠️ These symptoms may be serious. "
                "Please consider contacting a healthcare professional."
            )

        # STEP 6 — PREPARE EXPLAINABLE SOURCES
        sources = []
        for item in results[:3]:
            sources.append({
                "question": item.get("question", "")[:80],
                "source": item.get("source", "MedQuAD")
            })

        # STEP 7 — RETURN FINAL RESPONSE
        return jsonify({
            "answer": answer,
            "sources": sources
        })

    # ERROR HANDLING
    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


# RUN SERVER
if __name__ == "__main__":
    app.run(debug=True)
