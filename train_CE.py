import torch
import wandb
from tqdm import tqdm

def train(dataloader, model, loss_fn, optimizer, epoch, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    total_loss = 0.0

    tqdm_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")

    for X, y in tqdm_bar:
        X, y = X.to(device), y.long().to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = loss.item()
        total_loss += loss_value

        tqdm_bar.set_postfix({"loss": f"{loss_value:.6f}"})
        #wandb.log({"train_loss": loss_value, "epoch": epoch})

    avg_loss = total_loss / num_batches
    return avg_loss

def test(dataloader, model, loss_fn, epoch, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    tqdm_bar = tqdm(dataloader, total=len(dataloader), desc=f"Testing Epoch {epoch}")

    with torch.no_grad():
        for X, y in tqdm_bar:
            X, y = X.to(device), y.long().to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    # 최종 결과 출력
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    tqdm_bar.set_postfix({"test_loss": f"{test_loss:.6f}", "accuracy": f"{(100 * correct):>0.1f}%"})
    #wandb.log({"test_loss": test_loss, "test_accuracy": correct, "epoch": epoch})

    return test_loss, correct
