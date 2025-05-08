import torch

from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape


class SGMGNNRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        


        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        


        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        


        future_data, history_data, long_history_data = data
        history_data        = self.to_running_device(history_data)      # B, L, N, C
        long_history_data   = self.to_running_device(long_history_data)       # B, L, N, C
        future_data         = self.to_running_device(future_data)       # B, L, N, C

        history_data = self.select_input_features(history_data)
        long_history_data = self.select_input_features(long_history_data)


        prediction, pred_adj, prior_adj, gsl_coefficient = self.model(history_data=history_data, long_history_data=long_history_data, future_data=None, batch_seen=iter_num, epoch=epoch)

        batch_size, length, num_nodes, _ = future_data.shape
        assert list(prediction.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"


        prediction = self.select_target_features(prediction)
        real_value = self.select_target_features(future_data)
        return prediction, real_value, pred_adj, prior_adj, gsl_coefficient
