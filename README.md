# Aviation Safety Risk Estimation

A multi-task deep learning classifier that predicts aviation incident risk 
categories from free-text NASA ASRS reports and maps them to an ICAO 5×5 
safety risk matrix.

## Overview
- **Model:** Fine-tuned DistilBERT (multi-task classification)
- **Tasks:** 3 simultaneous predictions — Primary Problem, Events Anomaly, 
  Events Result
- **Risk Matrix:** ICAO Doc 9859 5×5 (Severity × Likelihood) with color-coded 
  output (green / yellow / orange / red)
- **Dataset:** ~111,000 NASA ASRS incident reports (2005–2025)

## Results
- Macro F1: ~0.277 (Primary Problem), ~0.277 (Events Anomaly), ~0.200 (Events Result)
- Known limitation: class imbalance across rare incident categories reduces 
  macro F1; frequent categories perform significantly better

## Deployed App
🤗 [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/MatheusHagemann/aviation-risk-estimator)

## Stack
Python · DistilBERT · HuggingFace Transformers · scikit-learn · pandas · Gradio
