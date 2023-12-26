# CUDA_VISIBLE_DEVICES=0 nohup python train_model.py -m TrGNN [-D sg_expressway_8weeks -p TrGNN_1581343606_100epoch.cpt -c 1] &
import pandas as pd
import time
from datetime import date, timedelta
from datetime import datetime as dt
import os
from dataset import Dataset
from utils import *
import random
import numpy as np
from math import radians, degrees, sin, cos, asin, acos, sqrt
import pickle as pkl
from metrics import *
from trajectory_transition import extract_trajectory_transition
from road_graph import extract_road_adj
from road import get_my_adj
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from road import MBR
import argparse

SEED = 20232023

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

RN_DIR="/home/mys/traj_flow/smallhahahaMTR_PORTOM"
#RN_DIR="/home/mys/traj_flow/chengdu/chengdu_hahahaMTR_PORTOM"

# Arguments
parser = argparse.ArgumentParser(description='train_model')
parser.add_argument('-m', '--model_name', help='TrGNN', required=True)
parser.add_argument('-D', '--dataset', help='sg_expressway_8weeks', default='sg_expressway_8weeks')
parser.add_argument('-p', '--pre_trained', help='pre-trained model path. E.g. TrGNN_1581343606_100epoch.cpt', default='')
parser.add_argument('-c', '--calibrate', help='flow calibration on a daily basis', default=1)
args = parser.parse_args()
model_name, dataset, model_path, calibrate = args.model_name, args.dataset, args.pre_trained, bool(args.calibrate)
model_save_path=f"./model/" + str(model_name) + time.strftime("%Y%m%d_%H%M%S") + "/"
os.mkdir(model_save_path)
batch_size = 32
print(f"the model path is {model_save_path}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

grid_size = 50
start_time = time.time()
rn = load_rn_shp(RN_DIR, is_directed=True)
#mbr = MBR(30.35, 104.00, 30.55, 104.30)
mbr = MBR(39.88, 116.33, 39.95, 116.45)
rn_grid = get_rn_grid(mbr, rn, grid_size)
#print(rn_grid[100])
g_total = get_my_adj().to(device)
print(g_total)
max_lat, max_lng = 39.967, 116.472
#max_lat, max_lng = 30.785, 104.166
grid_num = gps2grid(SPoint(max_lat, max_lng), mbr, grid_size)
grid_num = (grid_num[0] + 1, grid_num[1] + 1)
#print(f"The grid num is {grid_num}")


# Model and log
# models = {'TrGNN':Model_TrGNN, 'TrGNN-':Model_GNN}
# model = models[model_name](rn_grid, grid_num)
model = MYModel_TrGNN(g_total, rn_grid, grid_num)
if model_path == '': # if no pre-trained model path
    prefix = '%s_%s'%(model_name, int(start_time))
    checkpoint_epoch = -1
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
    prefix = '_'.join(model_path.split('_')[:2])
    checkpoint_epoch = int(model_path.split('_')[-1][:-9])
model_path = 'model/%s_%sepoch.cpt'%(prefix, '%d')
log_path = 'log/%s.log'%prefix

model2 = model.to(device)
print_log(device, log_path)


# Dataset
# 'sg_expressway_4weeks', 'sg_expressway_8weeks'
road_adj = extract_road_adj() # directed adj

if dataset == 'demo':
    start_date, end_date = '20121001', '20121031'
    calibrate = False
elif dataset == 'sg_expressway_8weeks':
    start_date, end_date = '20160314', '20160424' # train period + validation period
elif dataset == 'chengdu':
    start_date, end_date = '20140818', '20140828'
    calibrate = False
else:
    start_date, end_date = '20160401', '20160421' # train period + validation period
trajectory_transition = extract_trajectory_transition(start_date, end_date)
# smoothing with binary road_adj, in case no historical flow is recorded.
road_adj_mask = np.zeros(road_adj.shape)
road_adj_mask[road_adj > 0] = 1
np.fill_diagonal(road_adj_mask, 0)
for i in range(len(trajectory_transition)):
    trajectory_transition[i] = trajectory_transition[i] + road_adj_mask

if dataset == 'demo':
    start_date, end_date = '20121001', '20121031'
elif dataset == 'chengdu':
    start_date, end_date = '20140818', '20140828'
elif dataset == 'sg_expressway_8weeks':
    start_date, end_date = '20160314', '20160508' # train (5 weeks) + validation (1 week) + test (2 weeks)
else:
    start_date, end_date = '20160401', '20160428' # train + validation + test
dates = date_range(start_date, end_date)
flow_df = pd.concat([pd.read_csv('data/flow_%s_%s.csv'%(date, date), index_col=0) for date in dates])
flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)
# flow calibration on a daily basis
if calibrate:
    print_log('Calibrating flow...', log_path)
    trajectory_metadata = pd.read_csv('data/trajectory_metadata.csv') # read trajectory metadata
    multipliers = np.repeat(np.array(trajectory_metadata['vehicles'][0] / trajectory_metadata['vehicles']), 96)
    multipliers[multipliers==np.inf]=0
    flow_df = flow_df.mul(multipliers, axis=0)
# print_log(flow_df.shape, log_path)
# print_log('Total flow: %d'%(flow_df.sum().sum()), log_path)


if dataset == 'demo': # 20160314
    indices = {'train': list(range(1288)), # first 5 weeks 20160314-20160417 (24-1)*(60/15)*56
           'val': list(range(1288, 1932)), # 6th week 20160418-20160424 (24-1)*(60/15)*7
           'test': list(range(1932, 2576))} # 7th-8th weeks 20160425-20160508 (24-1)*(60/15)*14
    weekdays = np.array([  #0, 1, 2, 3, 4,
                     7, 8, 9, 10, 11, # PH: 25th May, Friday
                     14, 15, 16, 17, 18,
                     21, 22, 23, 24, 25,
                     28, 29, 30, 31]) # PH: 2nd May, Monday
elif dataset == 'sg_expressway_8weeks': # version 20160314-20160508
    indices = {'train': list(range(3220)), # first 5 weeks 20160314-20160417 (24-1)*(60/15)*56
               'val': list(range(3220, 3864)), # 6th week 20160418-20160424 (24-1)*(60/15)*7
               'test': list(range(3864, 5152))} # 7th-8th weeks 20160425-20160508 (24-1)*(60/15)*14
    # indices of weekdays (exclude weekends and PHs)
    weekdays = np.array([0, 1, 2, 3, 4, 
                         7, 8, 9, 10, # PH: 25th May, Friday
                         14, 15, 16, 17, 18,
                         21, 22, 23, 24, 25,
                         28, 29, 30, 31, 32, 
                         35, 36, 37, 38, 39,
                         42, 43, 44, 45, 46, 
                         50, 51, 52, 53]) # PH: 2nd May, Monday
elif dataset == 'chengdu':
    indices = {'train': list(range(644)), # first 5 weeks 20160314-20160417 (24-1)*(60/15)*56
               'val': list(range(644, 828)), # 6th week 20160418-20160424 (24-1)*(60/15)*7
               'test': list(range(828, 1012))} # 7th-8th weeks 20160425-20160508 (24-1)*(60/15)*14
    # indices of weekdays (exclude weekends and PHs)
    weekdays = np.array([0, 1, 2, 3, 4, 
                         7, 8, 9, 10])

else: # version 20160401-20160428
    indices = {'train': list(range(1288)), # first two weeks (24-1)*(60/15)*14
               'val': list(range(1288, 1932)), # third week (24-1)*(60/15)*7
               'test': list(range(1932, 2576))} # fourth week (24-1)*(60/15)*7


scaler = StandardScaler().fit(flow_df.iloc[indices['train'] + indices['val']].values) # normalize flow


# Train model
loss_fn = nn.MSELoss()
learning_rate = 0.0001
num_epochs = 100
min_mae = 100 # initialize
early_stop_threshold = 0.0003 # for val_mae
result_function = result_analysis2 if dataset == 'demo' else result_analysis3
#N_ROAD=4388
N_ROAD=2613


def validate(model, mode='val'):
    # mode: ['val', 'test']. Validate on validation set or test set.
    
    running_loss = 0
    n_samples = 0
    
    h_init = torch.zeros(5, N_ROAD, 1) # (gru_num_layers, n_road, hidden_size)
    h_init = h_init.to(device)
    
    Y_true = np.zeros((len(indices[mode]), N_ROAD)) # (n_sample, n_road)
    Y_pred = np.zeros((len(indices[mode]), N_ROAD))
    for i in indices[mode]:

        d = i // 92
        t = i % 92

        X = normalized_flows[d*96+t : d*96+t+4]
        T = tuple(transitions_ToD[t:t+4])
        # W passed to device already
        y_true = normalized_flows[d*96+t+4]
        print(f"The shape of T is {T.shape}")
        
        ToD = torch.from_numpy(np.eye(24)[np.full((N_ROAD), ((t+4) * 15 // 60) % 24)]).float().to(device) # one-hot encoding: hour of day. (n_road, 24)
        DoW = torch.from_numpy(np.full((N_ROAD, 1), int(d in weekdays))).float().to(device) # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        y_pred = model(X, T, W, h_init, W_norm, ToD, DoW)
        
        Y_true[n_samples] = flow_df.iloc[d*96+t+4].values

        mya = y_pred.detach().cpu().numpy().reshape(1,-1)
        # print(f"the shape of mya is {mya.shape}")
        Y_pred[n_samples] = scaler.inverse_transform(mya)
        
        loss = loss_fn(y_pred, y_true)
        loss.detach_()

        running_loss += loss.item()
        n_samples += 1
    
    Y_pred[Y_pred < 0] = 0 # correction for negative values
    #print(f"the shape of true {Y_true.shape}, pred {Y_pred.shape}")
    mae = MAE(Y_pred, Y_true, main_roads=False)
    mape = MAPE(Y_pred, Y_true, main_roads=False)
    rmse = RMSE(Y_pred, Y_true, main_roads=False)
    print_log('>> %s_loss: %.3f, MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mode, running_loss/n_samples, mae, mape, rmse), log_path)
    
    return running_loss/n_samples, Y_pred, Y_true, mae

def validate2(model, data_iter, mode):
    model.eval()
    h_init = None
    running_loss = 0
    n_samples = len(data_iter)

    total_mae = 0
    total_mape = 0
    total_rmse = 0
    my_Y_true = []
    my_Y_pred = []
    for i, batch in enumerate(data_iter):
        X, T, ToD, DoW, y_true, raw_true = batch

        X = X.to(device)
        T = T.to(device)
        ToD = ToD.float().to(device)
        DoW  = DoW.float().to(device)
        y_true = y_true.to(device)
        raw_true  = raw_true.numpy()
        y_pred = model(X, T, W, h_init, W_norm, ToD, DoW, constraint_mat)

        #print(f"The my_pred shape is {y_pred.shape}")
        my_pred = y_pred.detach().cpu().numpy()
        raw_pred = scaler.inverse_transform(my_pred)
        loss = loss_fn(y_pred, y_true)
        loss.detach_()
        running_loss += loss.item()

        #print(f"the type of raw {type(raw_pred)}, {type(raw_true)}")
        raw_pred[raw_pred < 0] = 0
        mae = MAE(raw_pred, raw_true, main_roads=False)
        mape = MAPE(raw_pred, raw_true, main_roads=False)
        rmse = RMSE(raw_pred, raw_true, main_roads=False)
        # print(f"The raw pred is {raw_pred.shape}, the type is {type(raw_pred)}")
        my_Y_true.append(raw_true)
        my_Y_pred.append(raw_pred)

        total_mae += mae
        total_mape += mape
        total_rmse += rmse
    print_log('>> %s_loss: %.3f, MAE: %.3f, MAPE: %.3f, RMSE: %.3f'%(mode, running_loss/n_samples, total_mae / n_samples, total_mape / n_samples, total_rmse / n_samples), log_path)
    if mode == "test":
        my_Y_true = np.concatenate(my_Y_true, axis=0)
        my_Y_pred = np.concatenate(my_Y_pred, axis=0)
        #print(f"The shape of my Y true and pred are {my_Y_true.shape}, {my_Y_pred.shape}")
        result_function(my_Y_pred, my_Y_true, model_type='ours', log_path=log_path) # result analysis on test results

    return running_loss / n_samples, total_mae / n_samples

# preprocessing
print_log('Preprocessing...', log_path)
normalized_flows = torch.from_numpy(scaler.transform(flow_df.values)).float() # for X. normalized
#transitions_ToD = torch.stack([to_sparse_tensor(normalize_adj(trajectory_transition[i])) for i in range(len(trajectory_transition))]) # for T. time of day
transitions_ToD = [to_sparse_tensor(normalize_adj(trajectory_transition[i])) for i in range(len(trajectory_transition))] # for T. time of day
#print(f"the shape of TOod is {transitions_ToD.shape}")
#torch.save(transitions_ToD, "./myTensor.pt")
# TODO
#transitions_ToD =  torch.load("./myTensor.pt")
W = torch.from_numpy(road_adj) # for W
W_norm = torch.from_numpy(normalize_adj(road_adj, mode='aggregation')).to(device) # for normalized W
print_log('Preprocessing completed. Clock: %.0f seconds'%(time.time() - start_time), log_path)
constraint_mat = get_constraint_mat(N_ROAD, rn).to(device)
#W_norm = constraint_mat.to(device)

# print(f"The shape of flows, transition, weekdays {normalized_flows.shape}, {transitions_ToD.shape}")
## todo train, val, test
train_dataset = Dataset(normalized_flows,flow_df, weekdays,transitions_ToD, indices, 'train')
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Dataset(normalized_flows,flow_df, weekdays,transitions_ToD, indices, 'val')
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Dataset(normalized_flows,flow_df, weekdays,transitions_ToD, indices, 'test')
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
stopping_count = 0 
for epoch in range(num_epochs):
    model2.train()
    h_init = torch.zeros(5, N_ROAD, 1) # (gru_num_layers, n_road, hidden_size)
    h_init = h_init.to(device)
    running_loss = 0
    n_samples = len(train_iter)

    for i, batch in enumerate(train_iter):
        X, T, ToD, DoW, y_true, _ = batch

        X = X.to(device)
        T = T.to(device)
        ToD = ToD.float().to(device)
        DoW  = DoW.float().to(device)
        y_true = y_true.to(device)
        #print(f"The shape of X, T, dow, y_true {X.shape}, {T.shape}, {DoW.shape}, {y_true.shape}")
        #print(f"the type of everything is {X.dtype}, {T.dtype}, {W_norm.dtype}, {ToD.dtype}, {DoW.dtype}")

        y_pred = model2(X, T, W, h_init, W_norm, ToD, DoW, constraint_mat)
        #print(f"The type of y_pred is {y_pred.dtype}")
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1)  # log_vars are not necessary to clip

        optimizer.step()
        
        running_loss += loss.item()
        if i % 200 == 0:
            print(running_loss/(i+1))

    print_log('Validating...', log_path)
    val_loss, val_mae = validate2(model2,val_iter, 'val')
    line = 'Epoch %d, time spent: %.0f seconds, train_loss: %.3f, val_loss: %.3f'%(epoch, time.time()-start_time, running_loss/n_samples, val_loss)
    print_log(line, log_path)
    if val_mae < min_mae:
        stopping_count = 0
        min_mae = val_mae
        print_log('Saving model...', log_path)
        torch.save(model2, model_save_path + 'val-best-model.pt')
        print_log('Saving results...', log_path)
        # with open('result/%s_Y_true.pkl'%(prefix), 'wb') as f:
            # pkl.dump(Y_true, f)
        # with open('result/%s_%sepoch_Y_pred.pkl'%(prefix, epoch), 'wb') as f:
            # pkl.dump(Y_pred, f)
#         result_function(Y_pred, Y_true, model_type='ours', log_path=log_path) # result analysis on test results
    else:
        stopping_count += 1
        
    if stopping_count >= 10:
        print_log('myEarly stop.', log_path)
        break
    



# print_log('Training model...', log_path)
# stopping_count = 0
# for epoch in range( num_epochs):
    # model.train()
    
    # print_log('Epoch %d'%epoch, log_path)
    # print("what the")
    
    # if epoch%30 == 0:
        # learning_rate /= 2
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # h_init = torch.zeros(5, N_ROAD, 1) # (gru_num_layers, n_road, hidden_size)
    # h_init = h_init.to(device)
    
    # running_loss = 0
    # n_samples = 0
    
    # np.random.shuffle(indices['train'])
    # for index in range(len(indices['train'])):
        
        # i = indices['train'][index]
        # d = i // 92
        # t = i % 92
        
        # X = normalized_flows[d*96+t : d*96+t+4] # tensor: (n_timestamp, n_road)
        # T = tuple(transitions_ToD[t:t+4]) # tuple of n_timestamp sparse_tensors: (n_road, n_road)
        # # W passed to device already # sparse_tensor: (n_road, n_road)
        # y_true = normalized_flows[d*96+t+4] # (n_road)
        
        # optimizer.zero_grad()
        # ToD = torch.from_numpy(np.eye(24)[np.full((N_ROAD), ((t+4) * 15 // 60) % 24)]).float().to(device) # one-hot encoding: hour of day. (n_road, 24)
        # DoW = torch.from_numpy(np.full((N_ROAD, 1), int(d in weekdays))).float().to(device) # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        # y_pred = model(X, T, W, h_init, W_norm, ToD, DoW)
        # loss = loss_fn(y_pred, y_true)
        # loss.backward()
        
        # optimizer.step()
        
        # running_loss += loss.item()
        # n_samples += 1
        # if n_samples % 500 == 0:
            # print_log('Epoch %d, %d samples, clock: %.0f seconds'%(epoch, n_samples, time.time() - start_time), log_path)
    
    # train_loss = running_loss/n_samples
    # print_log('Validating...', log_path)
    # val_loss, Y_pred, Y_true, val_mae = validate(model,mode='val')
    # line = 'Epoch %d, time spent: %.0f seconds, train_loss: %.3f, val_loss: %.3f'%(epoch, time.time()-start_time, train_loss, val_loss)
    # print_log(line, log_path)
    
    # if val_mae < min_mae:
        # stopping_count = 0
        # min_mae = val_mae
        # print_log('Saving model...', log_path)
        # # torch.save(model.state_dict(), model_path%epoch)
        # torch.save(model, model_save_path + 'val-best-model.pt')
        # print_log('Saving results...', log_path)
        # with open('result/%s_Y_true.pkl'%(prefix), 'wb') as f:
            # pkl.dump(Y_true, f)
        # with open('result/%s_%sepoch_Y_pred.pkl'%(prefix, epoch), 'wb') as f:
            # pkl.dump(Y_pred, f)
# #         result_function(Y_pred, Y_true, model_type='ours', log_path=log_path) # result analysis on test results
    # else:
        # stopping_count += 1
        
    # if stopping_count >= 10:
        # print_log('myEarly stop.', log_path)
        # break

    # if min_mae < early_stop_threshold:
        # print_log('Early stop.', log_path)
        # break

print_log('Testing...', log_path)
model2 = torch.load(model_save_path + 'val-best-model.pt').to(device)

test_loss, test_mae = validate2(model2, test_iter, mode='test')

