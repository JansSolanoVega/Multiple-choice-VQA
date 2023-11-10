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
print("text", text.shape)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    print(image_features.shape)
    print(text_features.shape)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    



    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    end = time.time()
print("Logits per image: ", logits_per_image)
print("Logits per text: ", logits_per_text)
print("Similarity: ", similarity)
print("Label probs:", probs, "time: ", end-start)  # prints: [[0.9927937  0.00421068 0.00299572]]