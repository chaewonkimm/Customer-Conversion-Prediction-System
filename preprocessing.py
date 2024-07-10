import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

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

        features = session_data[['hour', 'event_nm', 'ad_clicked_sequence', 'conversion_seconds', '누적_구매횟수',
                                '누적_상품클릭횟수', '누적_검색횟수', '누적_장바구니횟수', '관심도_점수']].to_numpy()
        features = features.astype(float)

        earlier_sessions = self.adid_to_sessions.get(adid, [])
        for esid in earlier_sessions:
            if esid == session_id:
                continue
            earlier_session_data = self.sessions[esid]
            if (start_time - earlier_session_data['date_time'].iloc[-1]).total_seconds() <= 86400:
                earlier_features = earlier_session_data[['hour', 'event_nm', 'ad_clicked_sequence', 'conversion_seconds', '누적_구매횟수',
                                '누적_상품클릭횟수', '누적_검색횟수', '누적_장바구니횟수', '관심도_점수']].to_numpy() * 0.5
                features = np.vstack([features, earlier_features])

        target = session_data.iloc[-1]['target2'] # target2 : 구매전환율, target3 : 이탈률
        
        ## Preprocessing data in units of seconds, minutes, and hours
        seconds_features = self._aggregate_by_time_unit(session_data, 'S')
        minutes_features = self._aggregate_by_time_unit(session_data, 'T')
        hours_features = self._aggregate_by_time_unit(session_data, 'H')

        return seconds_features, minutes_features, hours_features, target

    def _aggregate_by_time_unit(self, session_data, time_unit):
        if time_unit == 'S':
            session_data['time_unit'] = session_data['conversion_time'].dt.total_seconds().astype(int)
        elif time_unit == 'T':
            session_data['time_unit'] = (session_data['conversion_time'].dt.total_seconds() // 60).astype(int)
        elif time_unit == 'H':
            session_data['time_unit'] = (session_data['conversion_time'].dt.total_seconds() // 3600).astype(int)
        else:
            raise ValueError("Unsupported time unit")

        grouped = session_data.groupby('time_unit').mean(numeric_only=True).reset_index()
        features = grouped[['hour', 'event_nm', 'ad_clicked_sequence', 'conversion_seconds', '누적_구매횟수',
                            '누적_상품클릭횟수', '누적_검색횟수', '누적_장바구니횟수', '관심도_점수']].to_numpy()

        if len(features) == 0:
            features = np.zeros((1, 9))
        elif features.shape[1] != 9:
            raise ValueError(f"Feature size mismatch. Expected 9 but got {features.shape[1]}")
        
        hidden_dim = 64
        expected_dim = hidden_dim * 2
        padded_features = np.zeros((features.shape[0], expected_dim))
        padded_features[:, :features.shape[1]] = features

        return torch.tensor(padded_features, dtype=torch.float)

def load_data(file_path):
    dataset = SessionDataset(file_path)
    return dataset