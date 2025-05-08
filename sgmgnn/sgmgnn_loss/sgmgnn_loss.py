import numpy as np
from torch import nn
from basicts.losses import masked_mae

def sgmgnn_loss(prediction, real_value, theta, priori_adj, gsl_coefficient, null_val=np.nan):

    B, N, N = theta.shape
    theta = theta.view(B, N*N)
    tru = priori_adj.view(B, N*N)
    BCE_loss = nn.BCELoss()
    loss_graph = BCE_loss(theta, tru)

    loss_pred = masked_mae(preds=prediction, labels=real_value, null_val=null_val)

    loss = loss_pred + loss_graph * gsl_coefficient
    return loss
