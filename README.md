# Policy-LLM

# Compliance Checker API

## Introduction
I built this API to analyze the compliance of a webpage against a given policy document. The API takes in a webpage URL and a policy URL as input, processes the text, and determines which sections of the webpage violate the policy. The findings are returned as an array of non-compliant statements, making it easy to identify areas that need to be addressed.

The API is designed using **Fastify** for the Node.js server and **Flask** for the Python-based compliance checking. The reason behind this architecture is to leverage Fastify's performance benefits over Express while using Python for natural language processing and compliance evaluation. The system is containerized using **Docker** for seamless deployment.

## Task Overview
### Given Task:
The task was to:
- Build an API that takes a webpage and checks its content against a compliance policy.
- Return a list of non-compliant results.
- Use OpenAI or any open-source LLMs for compliance analysis.
- Since the company primarily works with **TypeScript**, implementing the API in **Node.js** was preferred, with minimal Python usage.

### Example Test Case:
- **Policy URL:** [Stripe's compliance policy](https://stripe.com/docs/treasury/marketing-treasury)
- **Webpage URL:** [Mercury's website](https://mercury.com/)
- The system should check if Mercury's website aligns with Stripe's compliance policy.

## Thought Process and Development Approach

### Step 1: Structuring the API
I started by designing the API to have a single endpoint that takes three parameters:
1. `webpageUrl` - The URL of the webpage to analyze.
2. `policyUrl` - The URL of the compliance policy.
3. `mode` - Whether to use Python-based compliance checking or Gemini API.

### Step 2: Choosing Fastify Over Express
Instead of using **Express.js**, I opted for **Fastify** due to its better performance and developer experience. Fastify allows for quick API responses, making it a good choice for a real-time compliance-checking service.

### Step 3: Implementing the API Route
I created a route in Fastify that:
- Validates the incoming request using middleware.
- Ensures `webpageUrl` and `policyUrl` are present.
- Calls the **controller function** to process the compliance check.

### Step 4: Extracting Webpage and Policy Text
I implemented a **scraper service** that fetches text from a webpage and cleans it:
- Uses `axios` to fetch the HTML.
- Uses `cheerio` to extract text from `<p>`, `<h1>`, `<h2>`, `<h3>`, `<li>` elements.
- Removes extra characters, URLs, and formatting issues to ensure clean text.

### Step 5: Implementing Compliance Analysis
For compliance analysis, I considered two approaches:
#### 1. **Python-Based Compliance Checker**
- Uses **spaCy** for NLP processing.
- Implements **MobileBERT** (typeform/mobilebert-uncased-mnli) for zero-shot classification.
- Splits policy text into clauses and compares webpage text to policy statements.
- Uses **Flask** to expose an endpoint for compliance checking.
- Runs the analysis in parallel using `ThreadPoolExecutor` to speed up processing.

#### 2. **OpenAI Gemini API**
- When `mode=gemini`, the API sends a request to the Gemini model.
- The model analyzes webpage text against policy text and returns JSON results.
- This approach is more scalable but comes with API costs.

### Step 6: Dockerizing the Application
Since I wanted to make the system **portable and easy to deploy**, I containerized the application using **Docker**:
- The Node.js Fastify server runs in one container.
- The Python compliance checker runs in another container.
- Both containers communicate using Docker Compose.

## API Endpoints
### 1. Compliance Check Endpoint
#### **POST** `/api_compliance_check`
##### Request Body:
```json
{
  "webpageUrl": "https://mercury.com/",
  "policyUrl": "https://stripe.com/docs/treasury/marketing-treasury",
  "mode": "gemini"
}
```
##### Response Example:
```json
{
  "webpageUrl": "https://mercury.com/",
  "policyUrl": "https://stripe.com/docs/treasury/marketing-treasury",
  "nonCompliantResults": [
    {
      "webpageSentence": "We offer free treasury services.",
      "policyClause": "Financial services must comply with regulatory guidelines.",
      "confidence": 0.85
    }
  ]
}
```

## Key Decisions and Learnings
1. **Why Fastify?**
   - It is significantly faster than Express.js.
   - Provides built-in schema validation and improved logging.
2. **Why Python for NLP?**
   - **Natural.js** is not as powerful for text classification.
   - **spaCy** and **transformers** provide better results.
3. **Why Docker?**
   - Ensures the API runs consistently across different environments.
   - Makes deployment easier for the interview.
4. **Tradeoff Between Gemini API and Python Model**
   - Gemini API provides better accuracy but costs money.
   - Python-based model is cost-effective but slower.
   - I implemented both options for flexibility.

## Deployment and Running Locally
### Running with Docker:
```sh
docker-compose up --build
```
This will spin up both Node.js and Python services.

### Running Manually:
1. Start the Python server:
```sh
cd python-server
python app.py
```
2. Start the Node.js server:
```sh
npm install
node server.js
```

## Final Thoughts
This was an interesting challenge because it combined multiple technologies and required a balance between cost, efficiency, and accuracy. I enjoyed implementing this API in **Fastify** instead of **Express**, as it gave me exposure to a newer framework. The compliance-checking logic using **zero-shot classification** in Python was also a learning experience, especially in optimizing performance.

This API is **scalable**, **efficient**, and can be easily integrated into a larger system for compliance monitoring. 🚀

