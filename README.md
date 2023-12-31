# TinyLM
A small PyTorch implementation of the Tiny Stories paper [Ronen Eldan and Yuanzhi Li, 2023].
The goal was to train a very small language model with the ability to generate basic content
on a limited domain. 

### Dataset
The following[Hugging Face dataset provided by the authors of TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) was used.

### Model
Models were trained on an RTX 3070Ti (8GB VRAM) GPU for an hour or two, which is enough to get the model to story generation capabilities on
the following hyperparameters:
* block size: 128
* embedding size: 768
* attention heads (per multi-head self-attention in decoder block): 4
* decoder blocks: 2
* vocab size: using vocab size of GPT2FastTokenizer from the **transformers** library
* initial learning rate: 0.0003
* batch size: 32

If you want to test the models, let me know and I will provide them via Hugging Face or Drive (or something else) since they are ~1.5GB in size still
without quantization.
