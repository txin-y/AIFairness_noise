import torch
import torchvision
from torchvision import transforms
from torchvision import models
import numpy as np
from tqdm import tqdm
from fair_metrics import compute_fairness_metrics
from fitzpatrick17k_data import fitzpatric17k_dataloader_score_v2

def set_noise(model, dev_var_std):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            scale = m.weight.abs().max().item()
            noise = torch.randn_like(m.weight) * scale * dev_var_std
            m.weight.data += noise.data
            
def inference(model, testloader, device="cpu"):
    # Something you need to copy from Yuanbo's code
    # The following lines are simple inference code for a model
    # It collects accuracy but you will also need a fairness score
    label_list = []
    y_pred_list = []
    skin_color_list = []
    ctype = "low" # low : num_classes == 144, medium, high
    f_attr = "skin_color_binary"
    
    for i, data in enumerate(tqdm(testloader)):
        inputs, labels = data["image"].float().to(device), torch.from_numpy(np.asarray(data[ctype])).long().to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        label_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(predicted.detach().cpu().numpy())
        skin_color_list.append(data[f_attr].numpy())
    
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    skin_color_list = np.concatenate(skin_color_list)

    fairness_metrics = compute_fairness_metrics(label_list[skin_color_list!=-1], y_pred_list[skin_color_list!=-1], skin_color_list[skin_color_list!=-1])
    return fairness_metrics

import pickle

BS = 128
NW = 0
num_classes = 114
 
trainloader, valloader, testloader, train_df = fitzpatric17k_dataloader_score_v2(BS, NW, predefined_root_dir='./fitzpatrick17k/dataset_images', csv_file_name='./fitzpatrick17k/fitzpatrick17k.csv', ctype='mid')
loaded_dict = torch.load("fitzpatrick17k_baseline.pth.tar") # You will need to put the trained model weight file from Yuanbo here

model = models.resnet18(num_classes=num_classes) # You will need to put the model used by Yuanbo here

NUMBER_OF_RUNS = 1000 # number of Monte Carlo runs
start_iteration = 542 # if you want to continue from a checkpoint, change this to the iteration number of the checkpoint
results_list = []
for i in range(start_iteration, NUMBER_OF_RUNS):
    model.load_state_dict(loaded_dict['state_dict'])
    set_noise(model, 0.1)
    results = inference(model, testloader)
    results_list.append(results)
    # Save results to file after every 100 runs
    if i % 500 == 0:
        print(i)
        with open(f'results_{i}.pkl', 'wb') as f:
            pickle.dump(results_list, f)
        results_list = []

            
# Save final results to file
with open('results.pkl', 'wb') as f:
    pickle.dump(results_list, f)
    

import pickle

with open('results.pkl', 'rb') as f:
    results_list = pickle.load(f)
    
# Continue running Monte Carlo simulations and appending to results_list

print(results_list[0])
print(results_list[1])
print(results_list[2])
print(len(results_list))
bin_num = len(results_list)

import matplotlib.pyplot as plt
import seaborn as sns

fairness_metrics = ['DP', 'EOpp0', 'EOpp1', 'EOdds', 'DP_abs', 'EOpp0_abs', 'EOpp1_abs', 'EOdds_abs', 'EOdds_new']

for metric in fairness_metrics:
    metric_hist = [r[f'fairness/{metric}'] for r in results_list]
    sns.kdeplot(metric_hist, color = 'darkblue')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of fairness/{metric}')
    plt.show()
    