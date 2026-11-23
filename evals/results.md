# Resume Tailor Agent - Evaluation Benchmark Results

Benchmark executed across **5 realistic candidate cases** with model `llama3.2:latest`.

## Summary Metrics Table

| Metric | Measured Value | Target / Benchmark |
| :--- | :--- | :--- |
| **Schema Validity Rate** | **0.0%** | > 95% |
| **Tool Calling Path Rate** | **100.0%** | Monitored |
| **Deterministic Fallback Rate** | **0.0%** | < 100% (Safety Net) |
| **Requirement Recall (Must-Haves)** | **100.0%** | > 80% |
| **Fabrication Rate (Verifier OFF)** | **0.0%** | Baseline |
| **Fabrication Rate (Verifier ON)** | **0.0%** | **0.0% (Intercepted)** |
| **Average Session Latency** | **42699.0 ms** | < 5000 ms |

## Per-Case Breakdown

| Case ID | Title | Path Used | Must-Have Recall | Rewrites Generated | Flagged Claims | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `case_01` | Backend SWE Intern (Python / FastAPI / Docker) | `tool_calling` | 100.0% | 0 | 0 | 29086.4 ms |
| `case_02` | AI Agent / RAG Engineer | `tool_calling` | 100.0% | 0 | 0 | 95811.0 ms |
| `case_03` | Data Infrastructure Engineer Intern | `tool_calling` | 100.0% | 0 | 0 | 31868.8 ms |
| `case_04` | Fullstack AI Engineer | `tool_calling` | 100.0% | 0 | 0 | 28831.7 ms |
| `case_05` | Junior MLOps / Model Serving Engineer | `tool_calling` | 100.0% | 0 | 0 | 27897.2 ms |

## Methodology & Notes
- **Small Local Model Reliability**: Small quantized local models occasionally hallucinate numbers or omit function calling schemas. The dual safeguard (deterministic fallback + regex/tech verifier) prevents broken runs and ensures 0% unchecked hallucinations.
- **Requirement Recall**: Computed by evaluating presence of expected domain topics in extracted requirements.
