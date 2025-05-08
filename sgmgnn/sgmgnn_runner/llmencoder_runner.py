import torch

from easytorch.utils.dist import master_only
from basicts.data.registry import SCALER_REGISTRY
from basicts.runners import BaseTimeSeriesForecastingRunner


class LLMEncoderRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
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
        


        future_data, history_data = data
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)


        reconstruction_masked_tokens, label_masked_tokens = self.model(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch)





        return reconstruction_masked_tokens, label_masked_tokens

    @torch.no_grad()
    @master_only
    def test(self):
        

        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)

            prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])

            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(prediction_rescaled, real_value_rescaled, null_val=self.null_val)
                self.update_epoch_meter("test_"+metric_name, metric_item.item())
