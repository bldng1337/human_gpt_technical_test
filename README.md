---
title: demo_human_gpt
app_file: interactive_test.py
sdk: gradio
sdk_version: 4.43.0
---
# HumanGPT Game Test
<a target="_blank" href="https://colab.research.google.com/github/bldng1337/human_gpt_technical_test/blob/main/chat_test.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This is a Test of the feasibility of letting an LLM generate the user part while the bot part is written by the user.

We won't instruct the language model to roleplay as a user. Instead, we'll instruct it to generate the bot's responses as it was trained to do. Then, we let the model complete the user text blocks. Since the model doesn't distinguish between writing bot or user parts, we should be able to leverage its full training instead of trying to get it to rp which it was not trained for. Should also make gaslighting/confusing the model harder as its not pretending to be a user but should belive it is.

## How to use
For the Notebook:
Press the "Open in Colab" button to open the notebook in Google Colab.
For the Gradio App:
Visit: https://bldng-demo-human-gpt.hf.space/

## TODO
- Make a chatwindow with panel to test the model interactively
- test multiple backs and forths

## Models to compare/try:
- https://huggingface.co/ArliAI/Llama-3.1-8B-ArliAI-RPMax-v1.1-GGUF
- https://huggingface.co/ArliAI/Phi-3.5-mini-3.8B-ArliAI-RPMax-v1.1-GGUF
- https://huggingface.co/google/gemma-2-2b-it-GGUF
