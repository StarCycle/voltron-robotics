import torch
from torchvision.io import read_image
from voltron import load
IMG_A, IMG_B = "examples/verification/img/peel-carrot-initial.png", "examples/verification/img/peel-carrot-final.png"
LANGUAGE = "peeling a carrot"

image_a, image_b = read_image(IMG_A), read_image(IMG_B)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load("v-gen", device=device, freeze=True)
dual_img = torch.stack([preprocess(image_a), preprocess(image_b)])[None, ...].to(device)
score = model.score(dual_img, [LANGUAGE])
print(score)
