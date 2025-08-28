import json, torch
from PIL import Image, ImageOps
from torchvision import transforms
import gradio as gr
from torch import nn

IMG_SIZE = 224
mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class InferenceModel(nn.Module):
    def __init__(self, scripted_path, labels_path="labels.json", device="cpu"):
        super().__init__()
        self.model = torch.jit.load(scripted_path, map_location=device).eval()
        with open(labels_path) as f:
            self.idx_to_class = {int(k): v for k, v in json.load(f).items()}
        self.device = device

    @torch.inference_mode()
    def predict(self, pil_img):
        # Ensure 3 channels (strip alpha)
        if pil_img.mode in ("RGBA","LA"):
            bg = Image.new("RGB", pil_img.size, (255,255,255))
            pil_img = Image.alpha_composite(bg, pil_img.convert("RGBA")) if pil_img.mode=="RGBA" else pil_img.convert("RGB")
        pil_img = pil_img.convert("RGB")
        x = preprocess(pil_img).unsqueeze(0).to(self.model.device if hasattr(self.model, 'device') else self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = float(probs.max()), int(probs.argmax())
        label = self.idx_to_class[pred_idx]

        # Optional reject-if-uncertain (uncomment to enable):
        # if conf < 0.55:
        #     return {"label":"unsure", "confidence": round(conf,4)}

        return {"label": label, "confidence": round(conf, 4)}

def launch(scripted_path="dogcat_scripted.pt"):
    model = InferenceModel(scripted_path)
    def _fn(img):
        out = model.predict(img)
        return f"{out['label']} (confidence {out['confidence']:.3f})"
    gr.Interface(fn=_fn, inputs=gr.Image(type="pil"), outputs="text",
                 title="Dog vs Cat Classifier",
                 description="Upload any photo; the model will say dog or cat, with confidence.").launch()

if __name__ == "__main__":
    # CLI usage:
    # python predict.py path/to.jpg
    import sys
    if len(sys.argv) > 1:
        im = Image.open(sys.argv[1])
        m = InferenceModel("dogcat_scripted.pt")
        print(m.predict(im))
    else:
        launch()
