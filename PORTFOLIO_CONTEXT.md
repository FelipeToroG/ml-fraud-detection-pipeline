# Portfolio Context

This project is Project 1 of my AI/ML engineering portfolio. It demonstrates production ML engineering fundamentals: cost-aware model selection, zero-leakage pipelines, threshold optimization, SHAP explainability, a FastAPI service, contract tests, and Docker deployment.

## Why this project remains in my portfolio

The fraud detection project showcases the ML engineering layer cleanly and in isolation. The dataset uses PCA-anonymized features, which makes it ideal for demonstrating the modeling pipeline itself without the noise of domain-specific feature engineering. Every engineering choice in the project is visible and defensible on its own terms: the cost matrix, the threshold sweep, the SHAP attribution path, and the zero-leakage pipeline construction.

## My current flagship work

My next project, the [AML Compliance Platform](https://github.com/FelipeToroG/aml-transaction-monitoring), extends production ML engineering with a second pillar: LLM-powered case adjudication. Every flagged transaction in the AML system produces both a risk score and a structured, evidence-bound case narrative that a compliance officer can review and escalate.

The AML project is a different class of system. It demonstrates:

- ML detection with a hybrid Isolation Forest and calibrated XGBoost ensemble
- LLM integration using Pydantic v2 structured outputs with discriminated unions
- Citation grounding so every claim in the narrative cites a specific transaction or feature value
- Refusal as a first-class output when evidence is insufficient
- Prompt versioning embedded in audit logs for historical reconstruction
- Langfuse tracing and Prometheus observability
- PSI drift detection and a fairness audit
- Regulatory mapping to BSA, FinCEN SAR requirements, FATF typologies, and SR 11-7

If you are evaluating my current senior-level capabilities, the AML platform is the more representative work. The fraud project remains published because it shows the engineering fundamentals on which the AML platform builds.

## Portfolio progression

1. **ML Fraud Detection Pipeline (this project).** Production ML engineering fundamentals.
2. **AML Compliance Platform.** ML and LLM integration with production observability.
3. **Contract Intelligence Agent.** Pure AI engineering: RAG, agents, tool use, cloud deployment.
4. **End-to-End Pipeline + Power BI Dashboard.** BI engineering foundations.
