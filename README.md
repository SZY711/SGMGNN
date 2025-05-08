# <div align="center"> A Semantically Guided Multimodal Graph Neural Network for Process Factor Forecasting of Industrial IoT Systems </div>
![Whole graph](https://github.com/user-attachments/assets/91b364df-3f87-45bd-89db-e5522202b30b)

## Requirements

This work is based on [BasicTS](https://github.com/zezhishao/BasicTS) with `easy-torch==1.2.10`. Other dependencies can be seen in `requirements.txt`.
## Train SGMGNN
1. Run `sgmgnn/LLMEncoder_DYG_wi.py` to perform embedding. Move the best checkpoints to `LLMEncoder_ckpt`
2. Run `sgmgnn/SGMGNN_DYG_wi.py` to perform semantically guided graph generation and prediction.
