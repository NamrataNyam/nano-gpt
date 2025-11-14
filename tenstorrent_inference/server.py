from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch_tt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Patch PyTorch to delegate supported ops to Tenstorrent
torch_tt.patch_pytorch()

app = FastAPI()

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Loading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("Compiling model for Tenstorrent...")
model = torch.compile(model, backend="tt")   # <-- This makes it use the NPU
model.eval()

class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 80

@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}
