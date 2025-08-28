import os, math, time, json, random
from pathlib import Path

import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 32
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5                # early stopping
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_datasets():
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]  # ImageNet stats

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full = datasets.ImageFolder(DATA_DIR)
    class_to_idx = full.class_to_idx
    print("Classes:", class_to_idx)  # {'cats':0, 'dogs':1} expected

    n = len(full)
    n_test = int(TEST_SPLIT * n)
    n_val = int(VAL_SPLIT * n)
    n_train = n - n_val - n_test

    # Stratified-ish split by shuffling within class folders is handled by ImageFolder order;
    # for large datasets this random split is fine. For small datasets, do your own stratified split.
    train_set, val_set, test_set = random_split(full, [n_train, n_val, n_test],
                                               generator=torch.Generator().manual_seed(SEED))
    # Patch transforms (ImageFolder holds a single transform)
    train_set.dataset.transform = train_tfms
    val_set.dataset.transform   = eval_tfms
    test_set.dataset.transform  = eval_tfms
    return train_set, val_set, test_set, class_to_idx

def build_model(num_classes=2):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False  # warmup: freeze backbone
    in_feats = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, num_classes)
    )
    return m

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, count = 0.0, 0, 0
    all_targets, all_preds = [], []
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device if device!="cpu" else "cpu", dtype=torch.float16, enabled=is_train):
            logits = model(x)
            loss = criterion(logits, y)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        count += x.size(0)
        all_targets.extend(y.tolist()); all_preds.extend(preds.tolist())

    return total_loss / count, correct / count, np.array(all_targets), np.array(all_preds)

def main():
    train_set, val_set, test_set, class_to_idx = get_datasets()
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val, best_path = float("inf"), "best_dogcat.pt"
    no_improve = 0

    # --- Warmup (frozen backbone)
    print("\nWarmup (frozen backbone)")
    for epoch in range(3):
        print(f"Epoch {epoch+1}/3")
        tr_loss, tr_acc, *_ = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, *_ = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save({"model": model.state_dict(),
                        "class_to_idx": class_to_idx}, best_path)
        else:
            no_improve += 1

    # --- Fine-tune (unfreeze some backbone)
    for p in model.features[-3:].parameters():  # last 3 blocks
        p.requires_grad = True

    print("\nFine-tuning")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        tr_loss, tr_acc, *_ = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, y_true, y_pred = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save({"model": model.state_dict(),
                        "class_to_idx": class_to_idx}, best_path)
            print("✓ Saved new best:", best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    # --- Evaluate on test set with best model
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"])
    test_loss, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion)
    print(f"\nTEST: loss {test_loss:.4f} acc {test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[idx_to_class[0], idx_to_class[1]]))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # --- Export TorchScript for easy serving
    model.eval()
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    scripted = torch.jit.trace(model, example)
    ts_path = "dogcat_scripted.pt"
    scripted.save(ts_path)
    with open("labels.json", "w") as f:
        json.dump({v:k for k,v in class_to_idx.items()}, f)
    print("Saved:", best_path, "and", ts_path, "and labels.json")

if __name__ == "__main__":
    main()
