import json
import re
from flask import Flask, request, jsonify
import spacy
from transformers import pipeline

# Load NLP model
print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")

# Load Zero-Shot Classifier for compliance checking
print("Loading zero-shot classifier model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def clean_text(text: str) -> str:
    """
    Cleans and preprocesses text for reliable analysis.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text

def analyze_compliance(webpage_text: str, policy_text: str):
    """
    Compares webpage text against the compliance policy.
    """
    cleaned_webpage = clean_text(webpage_text)
    cleaned_policy = clean_text(policy_text)
    
    # Extract sentences
    webpage_sentences = [sent.text.strip() for sent in nlp(cleaned_webpage).sents if sent.text.strip()]
    
    non_compliant_results = []

    for sentence in webpage_sentences:
        classification = classifier(
            sequences=f"Policy: {cleaned_policy}\nWeb content: {sentence}",
            candidate_labels=["compliant", "non-compliant"],
            hypothesis_template="This text is {} with the policy."
        )
        
        top_label = classification["labels"][0]
        top_score = classification["scores"][0]
        
        if top_label == "non-compliant" and top_score > 0.7:
            non_compliant_results.append({
                "sentence": sentence,
                "confidence": round(float(top_score), 3)
            })

    return non_compliant_results

app = Flask(__name__)

@app.route("/check_compliance", methods=["POST"])
def check_compliance():
    """
    API to validate compliance.
    """
    data = request.get_json(force=True)
    webpage_text = data.get("webpageText", "")
    policy_text = data.get("policyText", "")

    if not webpage_text or not policy_text:
        return jsonify({"error": "Both 'webpageText' and 'policyText' are required."}), 400

    findings = analyze_compliance(webpage_text, policy_text)
    return jsonify({"nonCompliantResults": findings})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
