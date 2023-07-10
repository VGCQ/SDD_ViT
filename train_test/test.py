import torch
from tqdm import tqdm

def test(model, test_loader, device, loss_fn, eval_accu, eval_losses):
    model.eval()
    
    running_loss=0
    correct=0
    total=0
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            images,labels=data[0].to(device),data[1].to(device)
            outputs=model(images)

            loss= loss_fn(outputs,labels)
            running_loss+=loss.item()
      
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss=running_loss/len(test_loader)
    accu=100.*correct/total
    
    eval_losses.append(test_loss)
    eval_accu.append(accu)
    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
    return(accu, test_loss)