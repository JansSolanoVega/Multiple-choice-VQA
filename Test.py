import torch
import clip
from PIL import Image
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

start = time.time()
image = preprocess(Image.open("CLIP.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    end = time.time()
print("Label probs:", probs, "time: ", end-start)  # prints: [[0.9927937  0.00421068 0.00299572]]