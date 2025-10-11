# ---- train_eval.py ----
import torch
import torch.nn as nn
from lr_scheduler import WarmupCosineCfg, WarmupCosineLRScheduler

def train_model(model,
                train_loader,
                epochs=20,
                base_lr=1e-2,           # peak LR (after warmup)
                weight_decay=0.0,
                warmup_steps=0,
                min_lr=0.0,
                device=None,
                print_every=1,
                optimizer_ctor=None):
    """
    Uses SGD + manual warmup+cosine LR schedule stepped each batch.
    Pass optimizer_ctor to override optimizer (e.g., lambda params: torch.optim.Adam(params, lr=1.0)).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Build optimizer (LR will be set by scheduler; give any placeholder LR)
    if optimizer_ctor is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optimizer_ctor(model.parameters())

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    sched = WarmupCosineLRScheduler(
        optimizer,
        WarmupCosineCfg(base_lr=base_lr, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=min_lr)
    )

    model.train()
    global_step = 0
    for epoch in range(1, epochs + 1):
        total_loss, total_correct, total_items = 0.0, 0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1).float()

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()       # update weights
            lr_now = sched.step()  # then advance LR schedule

            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).long()
                total_correct += (preds == yb.long()).sum().item()
                bs = yb.size(0)
                total_items += bs
                total_loss += loss.item() * bs

            global_step += 1

        if epoch % print_every == 0:
            print(
                f"Epoch {epoch:02d} | loss={total_loss/total_items:.4f} | acc={total_correct/total_items:.4f} | lr={lr_now:.6f}"
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

    return {"loss": total_loss / total_items, "acc": total_correct / total_items}

@torch.no_grad()
def predict_proba(model, X: torch.Tensor, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    logits = model(X.to(device).float())
    return torch.sigmoid(logits).cpu().squeeze(1)
