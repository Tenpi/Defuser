import flask               
from __main__ import app, socketio
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

tokenizer = None
model = None

@app.route("/generate-prompt", methods=["POST"])
def generate_prompt():
    global tokenizer
    global model
    data = flask.request.json
    prompt = data["prompt"] if "prompt" in data else "1girl"

    if not tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if not model:
        model = GPT2LMHeadModel.from_pretrained("FredZhang7/anime-anything-promptgen-v2")

    nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)
    outs = nlp(prompt, max_length=76, num_return_sequences=1, do_sample=True, repetition_penalty=1.2, temperature=0.7, top_k=4, early_stopping=True)

    return outs[0]["generated_text"]
