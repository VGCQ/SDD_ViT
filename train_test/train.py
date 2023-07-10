from tqdm import tqdm

def train(model, epoch, train_loader, device, loss_fn, optimizer, train_accu, train_losses):
    print('\nEpoch : %d'%epoch)
    
    model.train()
    
    running_loss=0
    correct=0
    total=0
    
    for data in tqdm(train_loader):
        
        inputs,labels=data[0].to(device),data[1].to(device)
        
        outputs=model(inputs)
        
        loss=loss_fn(outputs,labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_loss=running_loss/len(train_loader)
    accu=100.*correct/total
        
    train_accu.append(accu)
    train_losses.append(train_loss)
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
    return(accu, train_loss)