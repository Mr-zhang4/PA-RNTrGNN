import torch
import numpy as np

#N_ROAD=4388
N_ROAD=2613

class Dataset(torch.utils.data.Dataset):
    def __init__(self, flows, raw_flows,  weekdays,transitions_ToD, indices, mode):
        indices = indices[mode]
        self.flows = flows[indices[0]:indices[-1]]
        self.raw_flows = raw_flows[indices[0]:indices[-1]]
        self.transition = transitions_ToD
        self.weekdays = weekdays
        self.length = len(self.flows)
        self.mode = mode
        
    def __len__(self):
        days = self.length // 96
        remaing = self.length % 96 
        return self.length - days * 4  - 4 

    def __getitem__(self, index):
        d = index // 92
        t = index % 92
        #print(f"the index is {index}, len is {len(self.flows)}, d {d}, t {t}, larger {d*96+t+4}")

        X = self.flows[d*96+t: d*96+t+4]
        T = self.transition[t:t+4]
        # print(f"the t is {t}, {T.shape}")
        y_true = self.flows[d*96+t+4]
        T = torch.stack(T)
        ToD = np.eye(24)[np.full((N_ROAD), ((t+4) * 15 // 60) % 24)] # one-hot encoding: hour of day. (n_road, 24)
        DoW = np.full((N_ROAD, 1), int(d in self.weekdays)) # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        y_raw_true = self.raw_flows.iloc[d*96+t+4].values
        # print(f"the shape of everything {X.shape}, {T.shape}, {ToD.shape}, {DoW.shape}, {y_true.shape}, {y_raw_true.shape}")

        return X, T, ToD,DoW, y_true, y_raw_true
