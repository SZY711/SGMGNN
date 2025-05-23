import os

import torch
from torch.utils.data import Dataset
from basicts.utils import load_pkl


class ForecastingDataset(Dataset):
    

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, seq_len:int) -> None:
        

        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)

        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()

        self.index = load_pkl(index_file_path)[mode]

        self.seq_len = seq_len

        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        

        idx = list(self.index[index])

        history_data = self.data[idx[0]:idx[1]]     # 12
        future_data = self.data[idx[1]:idx[2]]      # 12
        if idx[1] - self.seq_len < 0:
            long_history_data = self.mask
        else:
            long_history_data = self.data[idx[1] - self.seq_len:idx[1]]     # 11

        return future_data, history_data, long_history_data

    def __len__(self):
        

        return len(self.index)
