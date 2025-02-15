import json
import re
import logging
from flask import Flask, request, jsonify
import spacy
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# --------------------------------------
# 1. Configuration & Logging
# --------------------------------------
# Set performance optimizations
torch.set_num_threads(4)  # Adjust if needed
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # More verbose logging

# --------------------------------------
# 2. Load Models
# --------------------------------------
logger.info("⚙️ Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
logger.info("✅ spaCy model loaded")

model_name = "typeform/mobilebert-uncased-mnli"  # 40x faster than BART for zero-shot
logger.info(f"⚙️ Initializing tokenizer/model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
logger.info(f"✅ Model loaded: {model_name}")

logger.info("⚙️ Setting up zero-shot pipeline...")
classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    max_length=512,  # Add truncation
    truncation=True
)
logger.info("✅ Zero-shot pipeline ready")

# --------------------------------------
# 3. Utility Functions
# --------------------------------------
def clean_text(text: str) -> str:
    """
    Perform minimal text cleaning while preserving 
    punctuation that might matter for sentence splitting.
    """
    logger.debug(f"🧹 Cleaning text sample: {text[:60]}...")
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra whitespace
    return text.lower()

def split_policy_clauses(policy_text: str) -> list[str]:
    """
    Attempt to split the policy into logical clauses 
    using bullet or numbered points. Otherwise fall back to sentence-level.
    """
    logger.info("📑 Splitting policy into clauses via bullet/numbered points...")
    
    # Try splitting on bullet points or "1. " patterns
    initial_clauses = re.split(r'\n\s*\d+\.\s+|\n\s*[-•*]\s+', policy_text)
    clauses = [c.strip() for c in initial_clauses if c.strip()]
    
    logger.debug(f"🔍 Found {len(clauses)} clauses from bullet/numbered patterns")

    # If we found too few clauses, fallback to sentence-level
    if len(clauses) < 2:
        logger.warning("⚠️ Using sentence splitting fallback for policy clauses")
        clauses = [sent.text.strip() for sent in nlp(policy_text).sents if sent.text.strip()]
        logger.debug(f"🔍 Fallback split => {len(clauses)} sentences")

    logger.info(f"📄 Final policy clauses count: {len(clauses)}")
    return clauses

# --------------------------------------
# 4. Core Compliance Function
# --------------------------------------
def analyze_compliance(webpage_text: str, policy_text: str):
    """Perform compliance checks in parallel, returning flagged results."""
    logger.info("🔎 Starting compliance analysis...")

    # -- Step A: Process Policy
    logger.debug("📝 Splitting policy text into clauses...")
    raw_clauses = split_policy_clauses(policy_text)
    policy_clauses = [clean_text(c) for c in raw_clauses if c.strip()]
    total_clauses = len(policy_clauses)

    # -- Step B: Process Webpage Text
    logger.debug("🌐 Processing webpage text for sentence splitting...")
    cleaned_webpage = clean_text(webpage_text)
    webpage_doc = nlp(cleaned_webpage)
    webpage_sents = [sent.text.strip() for sent in webpage_doc.sents if sent.text.strip()]

    logger.info(f"📊 Analyzing {len(webpage_sents)} webpage sentences vs {total_clauses} policy clauses")

    # Batching / parallel config
    batch_size = 16
    results = []

    def process_batch(batch_sentences):
        """
        Classify a batch of webpage sentences in the zero-shot pipeline 
        and log each classification outcome.
        """
        try:
            # 'batch_sentences' is a list of strings
            # We pass them to the pipeline with 'policy_clauses' as candidate labels.
            classifications = classifier(
                batch_sentences,
                policy_clauses,
                hypothesis_template="This violates: {}",
                multi_label=False,
                batch_size=batch_size
            )

            batch_results = []
            for i, classification in enumerate(classifications):
                top_label = classification["labels"][0]
                top_score = classification["scores"][0]
                sentence_text = batch_sentences[i]

                logger.debug(
                    f"🔎 Sentence: '{sentence_text[:50]}...' => "
                    f"Top Label: '{top_label}' (score: {top_score:.4f})"
                )

                # If confidence is above threshold, consider it non-compliant
                if top_score > 0.6:
                    # Map the label back to original clause text for clarity
                    original_clause = raw_clauses[policy_clauses.index(top_label)]
                    batch_results.append({
                        "webpageSentence": sentence_text,
                        "policyClause": original_clause,
                        "confidence": round(top_score, 3)
                    })
            return batch_results

        except Exception as e:
            logger.error(f"⚠️ Error processing batch: {str(e)}")
            return []

    # -- Step C: Parallel Execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for batch_idx in range(0, len(webpage_sents), batch_size):
            sub_batch = webpage_sents[batch_idx : batch_idx + batch_size]
            futures.append(executor.submit(process_batch, sub_batch))

        # Collect results as they complete
        for future in as_completed(futures):
            results.extend(future.result())

    logger.info(f"✅ Analysis complete. Found {len(results)} non-compliant items")
    return results, total_clauses

# --------------------------------------
# 5. Flask API
# --------------------------------------
app = Flask(__name__)

@app.route("/check_compliance", methods=["POST"])
def compliance_handler():
    logger.info("📥 Received new compliance request")
    try:
        data = request.get_json(force=True)
        if not data:
            logger.error("❌ Invalid JSON payload")
            return jsonify({"error": "Invalid JSON format"}), 400

        webpage_text = data.get("webpageText", "")
        policy_text = data.get("policyText", "")

        logger.debug(f"📦 Request data sizes => Webpage: {len(webpage_text)} chars, Policy: {len(policy_text)} chars")

        if not webpage_text or not policy_text:
            logger.warning("⚠️ Missing required 'webpageText' or 'policyText'")
            return jsonify({"error": "Missing required fields"}), 400

        # Run analysis
        findings, total_clauses = analyze_compliance(webpage_text, policy_text)

        return jsonify({
            "nonCompliantResults": findings,
            "clausesChecked": total_clauses
        })

    except Exception as e:
        logger.error(f"🔥 Critical error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("🚀 Starting compliance API on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
