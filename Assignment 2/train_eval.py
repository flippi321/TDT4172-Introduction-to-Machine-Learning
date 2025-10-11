import torch
import torch.nn as nn

def train_model(model,
                train_loader,
                epochs=20,
                lr=1e-3,
                weight_decay=0.0,
                device=None,
                print_every=1):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, total_correct, total_items = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1).float()  # shape (N, 1)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).long()
                total_correct += (preds == yb.long()).sum().item()
                total_items += yb.size(0)
                total_loss += loss.item() * yb.size(0)

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch:02d} | loss={total_loss/total_items:.4f} | acc={total_correct/total_items:.4f}"
            )

@torch.no_grad()
def evaluate(model, data_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_correct, total_items = 0.0, 0, 0

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1).float()

        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        total_correct += (preds == yb.long()).sum().item()
        total_items += yb.size(0)
        total_loss += loss.item() * yb.size(0)

    avg_loss = total_loss / total_items
    acc = total_correct / total_items
    return {"loss": avg_loss, "acc": acc}

@torch.no_grad()
def predict_proba(model, X: torch.Tensor, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    logits = model(X.to(device).float())
    return torch.sigmoid(logits).cpu().squeeze(1)
