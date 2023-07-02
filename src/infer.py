from PIL import Image
import matplotlib.pyplot as plt
import torch
    
def infer(model, image_path, device='cpu'):
    model.to(device)
    model.eval()

    with Image.open(image_path) as image:
        image_tensor = model.transforms(image).unsqueeze(0).to(device)
        plt.imshow(image)

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        probs = probs.squeeze()

    plt.title(f"Predicted label: {pred.item()}\nProbability: {100 * probs[pred.item()]:.1f}%")
    plt.axis("off")
    plt.show()
