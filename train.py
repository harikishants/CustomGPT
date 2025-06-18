import torch
import json
from tqdm import tqdm
from inference import generate_text

def train_gpt(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path='gpt_model.pt', start_epoch=1, patience=3, scheduler=None):

    model.to(device)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    epoch_no_improve = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            inputs = batch["input_ids"].to(device) # (B, T)
            labels = batch["labels"].to(device) # (B, T)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(
                            logits.reshape(-1, logits.size(-1)), # (B, T, V) --> (B*T, V)
                            labels.reshape(-1) # (B, T) --> (B*T, )
                            )
        
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate_gpt(model, val_loader, device, criterion)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"[info] Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if epoch % 10 == 0:
            path = save_path[:-3] + f'_epoch_{epoch}_loss_{avg_val_loss:.4f}.pt'
            save_model(model, optimizer, epoch, avg_val_loss, path)
            t = "chatbot"
            q = "Why are passengers requested to close their window blinds during a night-time takeoff?"
            a = ""
            prompt = f"<|task|>{t}\n<|question|>{q}\n<|answer|>{a}"
            output_text = generate_text(prompt, model=model, max_new_tokens=50, temperature=0.7, sampling='prob')
            print(output_text)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epoch_no_improve = 0
            path = save_path[:-3] + f'_best.pt'
            save_model(model, optimizer, epoch, avg_val_loss, path)
        else:
            epoch_no_improve += 1

        if epoch_no_improve >= patience:
            print(f"[info] Early stopping since there is no improvement.")
            break

        with open("training_history_gpt.json", "w") as f:
            json.dump(history, f)

@torch.no_grad()
def evaluate_gpt(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0

    for batch in val_loader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(inputs)
        loss = criterion(
                        logits.reshape(-1, logits.size(-1)), # (B, T, V) --> (B*T, V)
                        labels.reshape(-1) # (B, T) --> (B*T, )
                        )

        total_loss += loss.item()

    return total_loss / len(val_loader)

def save_model(model, optimizer, epoch, val_loss, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'hyperparams': model.hparams,
    }
    # path = path[:-3] + f'_{val_loss:.4f}_' + path[-3:]
    torch.save(checkpoint, path)

def load_gpt(model_class, optimizer_class, file_path, device=torch.device("cpu")):
    checkpoint = torch.load(file_path, map_location=device)

    model = model_class(**checkpoint['hyperparams'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = optimizer_class(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"[info] Loaded model from {file_path}")

    return model, optimizer, checkpoint['epoch'], checkpoint['val_loss'], checkpoint['hyperparams']




