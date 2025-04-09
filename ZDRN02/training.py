import torch


# 训练模型
def train_model(model, train_loader, criterion, optimizer, scheduler, device, l1_lambda, gradient_clip=1.0):
    model.train()
    running_loss = 0.0
    valid_batches = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.size(0) <= 1:
            continue
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        loss = loss + l1_lambda * l1_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        valid_batches += 1
    if valid_batches == 0:
        return 0
    return running_loss / valid_batches

