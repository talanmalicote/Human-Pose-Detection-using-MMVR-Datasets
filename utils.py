def compute_pck(preds, targets, threshold=0.05):
    preds = preds.view(-1,17,2)
    targets = targets.view(-1,17,2)

    dist = torch.norm(preds-targets, dim=2)
    correct = (dist < threshold).float()

    return correct.mean().item()

def train_and_evaluate(model, loader, epochs):
    history = {'loss': [], 'pck': []}

    for epoch in range(epochs):
        
      # Training phase
        model.train()
        total_loss = 0
        for radar, keypoints in loader:
            radar = radar.unsqueeze(1).to(device)  # Add channel dim
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            preds = model(radar)
            loss = criterion(preds, keypoints)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        history['loss'].append(avg_loss)

        # Evaluation phase
        model.eval()
        total_pck = 0
        with torch.no_grad():
            for radar, keypoints in loader:
                radar = radar.unsqueeze(1).to(device)
                keypoints = keypoints.to(device)

                preds = model(radar)
                pck = compute_pck(preds.cpu(), keypoints.cpu()) # PCK expects CPU tensors
                total_pck += pck
        avg_pck = total_pck / len(loader)
        history['pck'].append(avg_pck)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, PCK: {avg_pck:.4f}")

    return history
