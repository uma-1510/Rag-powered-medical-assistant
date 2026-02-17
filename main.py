from utils.retriever import retrieve, rerank_cross_encoder,rag_fusion_search
from utils.prompt_builder import build_gemini_prompt
from utils.retriever import print_retrieved_oneliners
from utils.gemini import call_gemini, get_truncated_gemini_answer
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json() or {}
    query = data.get('question', '').strip()

    top_k = 10

    try:
        def llm_generate_fn(prompt):
            lm_prompt = f"Given the question: '{prompt}', generate 3 relevant follow-up or reworded questions, one per line."
            response_text = call_gemini(lm_prompt, stream=False)  # Get full response, don't stream for simple QA
            questions = [q.strip("-• \n") for q in response_text.split("\n") if q.strip()]
            print("LLM generated questions:", questions)
            return questions

        # Use RAG Fusion to retrieve fused and reranked results
        fused_candidates = rag_fusion_search(query, llm_generate_fn, k_per_query=5, top_k=10)

        print_retrieved_oneliners(fused_candidates, max_items=3, maxlen=80)
        # Step 1: Retrieve candidates
        results = retrieve(query, k=top_k)

        # Optional: print to console for debugging (comment out if unwanted)
        print_retrieved_oneliners(results, max_items=3, maxlen=80)

        # Step 2: Rerank candidates
        reranked = rerank_cross_encoder(query, results)

        # Step 3: Build Gemini prompt
        gemini_prompt = build_gemini_prompt(query, reranked, max_contexts=3)

        # Step 4: Call Gemini and stream the answer
        answer_stream = call_gemini(gemini_prompt, stream=True)
        truncated_answer = get_truncated_gemini_answer(answer_stream, max_words=40)

        # Step 5: Prepare sources for output
        sources = []
        for item in reranked[:3]:
            sources.append({
                "question": item.get("question", "")[:60] + ("..." if len(item.get("question", "")) > 60 else ""),
                "url": item.get("url", "N/A")
            })

        # Return JSON response
        return jsonify({
            # "answer": truncated_answer,
            "sources": sources
        })

    except Exception as e:
        # Return error info in JSON, ideally log error for debug too
       return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


if __name__ == "__main__":
    # Run Flask app with debug=true for development
    app.run(debug=True)