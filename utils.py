import torch
from sklearn import metrics
from fair_metrics import compute_fairness_metrics
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# import tracemalloc

# fitz17k settings
# f_attr = "skin_color_binary"
ctype = 'low'

def NEachEval(model, device, test_loader, dev_var):
    """
    Quick evaluation function used in training. Not very precise.
    Returns evaluation accuracy.
    """
    model.eval()

    label_list = []
    y_pred_list = []
    skin_color_list = []
    total_samples = 0
    correct_predictions = 0
    # with tqdm(test_loader, dynamic_ncols=True) as tqdmDataLoader:
    # model.clear_noise()
    # with torch.no_grad():
        # for batch in tqdmDataLoader:
        #     images = batch["image"].float().to(device)
        #     labels = batch["label"].long().to(device)
        #     outputs = model(images)
    for _, data in enumerate(tqdm(test_loader)):
        model.clear_noise()
        model.set_noise(dev_var)

        inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        label_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(predicted.detach().cpu().numpy())
        skin_color_list.append(data["skin_color_binary"].numpy())
    
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    skin_color_list = np.concatenate(skin_color_list)
        
    overall_acc = metrics.accuracy_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1])

    return overall_acc

def NEval(model, device, test_loader, dev_var):
    """
        Slow evaluation function used in MC.
        returns evaluation accuracy
    """
    label_list = []
    y_pred_list = []
    skin_color_list = []
    # with tqdm(test_loader, dynamic_ncols=True) as tqdmDataLoader:
    # model, _, _, _, device, _, testloader = model_group
    model.eval()
    model.clear_noise()
    with torch.no_grad():
        model.set_noise(dev_var) 
        # Set noise once for one epoch, so you need to run this 
        # function multiple times, so it's slow
        for _, data in enumerate(tqdm(test_loader)):
            inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
            outputs = model(inputs)
            
        # for batch in tqdmDataLoader:
        #     images = batch["image"].float().to(device)
        #     labels = batch["label"].long().to(device)
        #     outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            label_list.append(labels.detach().cpu().numpy())
            y_pred_list.append(predicted.detach().cpu().numpy())
            skin_color_list.append(data["skin_color_binary"].numpy())
        
        label_list = np.concatenate(label_list)
        y_pred_list = np.concatenate(y_pred_list)
        skin_color_list = np.concatenate(skin_color_list)
        
    # accuracy = calculate_accuracy(label_list,y_pred_list,skin_color_list)
    accuracy = {
        'skin_color/overall_acc': metrics.accuracy_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1]),
        'skin_color/light_acc': metrics.accuracy_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0]),
        'skin_color/dark_acc': metrics.accuracy_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1]),
        'skin_color/overall_precision': metrics.precision_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        'skin_color/light_precision': metrics.precision_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        'skin_color/dark_precision': metrics.precision_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
        'skin_color/overall_recall': metrics.recall_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        'skin_color/light_recall': metrics.recall_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        'skin_color/dark_recall': metrics.recall_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
        'skin_color/overall_f1_score': metrics.f1_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], average='macro', zero_division=0),
        'skin_color/light_f1_score': metrics.f1_score(label_list[skin_color_list==0], y_pred_list[skin_color_list==0], average='macro', zero_division=0),
        'skin_color/dark_f1_score': metrics.f1_score(label_list[skin_color_list==1], y_pred_list[skin_color_list==1], average='macro', zero_division=0),
    }  
    fairness = compute_fairness_metrics(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], skin_color_list[skin_color_list!=-1])

    return accuracy, fairness

def NTrain(model_group, epochs, dev_var=0.0, verbose=False):
    """
        Training with noise injection
        model group: packed things needed for neural network training
        neural networks
        # header: file name for saving checkpoints
        dev_var: device value variation's standard deviation
        verbose: if you want to print out some information
        returns nothing
    """
    train_acc_list = []
    accuracy_list = []
    loss_list = []
    min_loss = 1000000
    
    model, criteria, optimizer, scheduler, device, trainloader, testloader = model_group
    print(f"Training with noise injection : {dev_var}")
    best_acc = 0.0

    for i in range(1,epochs+1):
        
    # with tqdm(trainloader, dynamic_ncols=False) as tqdmDataLoader:
        
        model.train()
        running_loss = 0.
        label_list = []
        y_pred_list = []
        skin_color_list = []
    
        # for batch in tqdmDataLoader:
        for _, data in enumerate(tqdm(trainloader)):
        
            model.set_noise(dev_var) # inject noise during training
            
            inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
            # images = batch["image"].float().to(device)
            # labels = batch["label"].long().to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # outputs = model(images)
            loss = criteria(outputs,labels)
            loss.backward()
            model.clear_noise()
            
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            label_list.append(labels.detach().cpu().numpy())
            y_pred_list.append(predicted.detach().cpu().numpy())
            skin_color_list.append(data["skin_color_binary"].numpy())
            # tqdmDataLoader.set_postfix(ordered_dict={
            #         "lr": optimizer.state_dict()['param_groups'][0]["lr"]
            #     })
            # data.set_postfix(loss=loss.item())
            
        label_list = np.concatenate(label_list)
        y_pred_list = np.concatenate(y_pred_list)
        skin_color_list = np.concatenate(skin_color_list)

        train_acc = metrics.accuracy_score(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1])
        overall_acc = NEachEval(model, device, testloader, dev_var)
    
        train_acc_list.append(train_acc)
        accuracy_list.append(overall_acc)
        loss_list.append(running_loss / len(trainloader))    

        if running_loss / len(trainloader) < min_loss: # save checkpoint if accuracy is higher than before
            min_loss = running_loss / len(trainloader)
            torch.save(model.state_dict(), f"vgg11_{dev_var}.pt")
            file_path = f"train_acc_{epochs}.txt"
            with open(file_path, "w") as file:
                file.write(f"epoch: {i}\n acc: {overall_acc}\n")
        if verbose: # print some training information if you want to
            print(f"epoch: {i:-3d}, train acc: {train_acc:.4f}, test acc: {overall_acc:.4f}, loss: {running_loss / len(trainloader):.4f}")
        scheduler.step()
        
    plt.plot(range(1,epochs+1), train_acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Train Accuracy Curve : {epochs}')
    plt.legend()
    plt.show()
    plt.savefig(f'train_acc_{epochs}_{dev_var}.png')
    
    plt.clf()
    
    plt.plot(range(1,epochs+1), accuracy_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Test Accuracy Curve : {epochs}')
    plt.legend()
    plt.show()
    plt.savefig(f'test_acc_{epochs}_{dev_var}.png')
    
    plt.clf()
    
    
    # Plotting the loss curve
    plt.plot(range(1,epochs+1), loss_list, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve : {epochs}')
    plt.legend()
    plt.show()
    plt.savefig(f'loss_{epochs}_{dev_var}.png')
        