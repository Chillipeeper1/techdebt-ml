---
title: TechDebt ML
emoji: 🧹
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Predicts technical debt risk per file in Python repos
models:
  - sentence-transformers/all-MiniLM-L6-v2
datasets:
  - ChilliPeep/techdebt-ml-dataset
---

# TechDebt ML — demo

Paste a public GitHub repo with Python code and get, per file, the predicted probability
that it carries technical debt.

Source code, training notebook and known limitations:
[github.com/Chillipeeper1/techdebt-ml](https://github.com/Chillipeeper1/techdebt-ml)
