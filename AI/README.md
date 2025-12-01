## AI Reasoning Engine (Person 2 – The Brainiac) 🤖

This folder contains the **intelligence layer** of the Health Data Analysis Agent.  
It takes preprocessed health data (from Python/pandas) and turns it into **clear, human‑friendly insights, warnings, and lifestyle suggestions**.

---

### Core Modules

- **`ai_reasoning_engine.py`**
  - `AIReasoningEngine`: Main orchestrator for rule‑based analysis.
  - Uses:
    - `HealthInsightGenerator` for core insights on:
      - Sleep (irregular / insufficient / good)
      - Heart rate (spikes, normal patterns)
      - Hydration (below / at recommended)
    - `HealthRecommendationEngine` for follow‑up actions based on severity.
    - `ContextualReasoning` to restrict analysis to the last **7 days** and compare to historical baselines.
    - `AdvancedPatternAnalyzer` (from `pattern_analyzer.py`) for richer anomaly detection.
  - **Output schema** (each insight is a dict with a consistent shape):
    - `id`: stable identifier, e.g. `"sleep:insufficient"`
    - `type`: `"sleep" | "heart_rate" | "hydration" | "activity"`
    - `severity`: `"info" | "good" | "caution" | "warning" | "critical"`
    - `title`: short human‑readable headline
    - `message`: full explanation in plain language
    - `suggested_actions`: list of strings (optional)
    - `metadata`: extra numeric/details for charts and debugging

- **`pattern_analyzer.py`**
  - `AdvancedPatternAnalyzer`: focuses on **anomaly and pattern detection**.
  - Sleep:
    - Weekend vs weekday differences
    - Improving/declining sleep duration trends
    - Irregularity based on variance
  - Heart rate:
    - Statistical outliers via IQR
    - Night‑time vs day‑time patterns
    - Simple risk level (`low`, `moderate`, `high`)
  - Its findings are converted into extra insights by `AIReasoningEngine`.

---

### LLM / RAG Components

- **`llm_insight_generator.py`**
  - `LLMInsightGenerator`:
    - Accepts aggregated **sleep** and **heart rate** statistics (already computed by pandas).
    - Builds prompts that:
      - Use a friendly system instruction (no diagnosis, encourage professional advice).
      - Include the user’s stats for the last days.
      - Include **retrieved public health guideline snippets** (RAG style).
    - Currently `_simulate_llm_response` fakes an LLM call with deterministic logic, but the prompt strings show how a real LLM would be wired.
  - `PublicHealthGuidelineRetriever`:
    - Tiny, in‑memory **RAG‑like retriever**.
    - Returns short guideline summaries (sleep, heart rate, hydration) plus a source note.
    - In a real deployment, this would query a vector store over public health papers/guidelines.
  - `EnhancedAIReasoningEngine`:
    - Wraps classic `AIReasoningEngine` logic with LLM‑style insight text.
    - Produces:
      - A list of LLM‑generated insights (sentences)
      - Lifestyle recommendations based on the combined insights

---

### API & Usage

- **`api_integration.py`**
  - Simple **Flask API** exposing:
    - `POST /analyze` – send JSON data for `sleep`, `heart_rate`, `hydration`.
    - Returns:
      - `insights`: list of structured insight objects (see schema above)
      - `recommendations`: unique list of suggested actions
      - `analysis_date`, `context_period`
    - This is the main entry point for other services / frontends.

- **`sample_usage.py` & `enhanced_sample.py`**
  - `sample_usage.py`:
    - Generates fake pandas DataFrames for sleep, heart rate, hydration.
    - Runs `AIReasoningEngine.analyze_health_data`.
    - Prints a **console report** with emojis and human‑friendly messages.
  - `enhanced_sample.py`:
    - Uses `EnhancedAIReasoningEngine` (LLM + RAG flavoured).
    - Demonstrates how to obtain:
      - LLM‑style insight sentences.
      - Lifestyle recommendations derived from those insights.


