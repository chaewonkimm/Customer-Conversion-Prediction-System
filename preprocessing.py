import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class SessionDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data['date_time'] = pd.to_datetime(self.data['date_time'])
        self.data['conversion_time'] = pd.to_timedelta(self.data['conversion_time'])
        self.data['conversion_seconds'] = self.data['conversion_time'].dt.total_seconds()
        self.data['event_nm'] = self.data['event_nm'].astype('category').cat.codes
        self.data.sort_values(by=['adid', 'date_time'], inplace=True)     

        self.sessions = {k: g for k, g in self.data.groupby('session')}
        self.adid_to_sessions = self.data.groupby('adid')['session'].unique()
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session_keys = list(self.sessions.keys())
        session_id = session_keys[idx]
        session_data = self.sessions[session_id]
        adid = session_data['adid'].iloc[0]
        start_time = session_data['date_time'].iloc[0]

        features = session_data[['hour', 'event_nm', 'ad_clicked_sequence', 'conversion_seconds']].to_numpy()
        features = features.astype(float)
        
        earlier_sessions = self.adid_to_sessions.get(adid, [])
        for esid in earlier_sessions:
            if esid == session_id:
                continue
            earlier_session_data = self.sessions[esid]
            if (start_time - earlier_session_data['date_time'].iloc[-1]).total_seconds() <= 86400:
                earlier_features = earlier_session_data[['hour', 'event_nm', 'ad_clicked_sequence', 'conversion_seconds']].to_numpy() * 0.5
                features = np.vstack([features, earlier_features])

        target = session_data.iloc[-1]['target'] - 1
        features_tensor = torch.tensor(features, dtype=torch.float)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return features_tensor, target_tensor

def load_data(file_path):
    dataset = SessionDataset(file_path)
    return dataset
