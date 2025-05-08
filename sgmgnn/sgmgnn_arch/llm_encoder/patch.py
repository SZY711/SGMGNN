from torch import nn


class PatchEmbedding(nn.Module):
    

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size             # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
                                        in_channel,
                                        embed_dim,
                                        kernel_size=(self.len_patch, 1),
                                        stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        

        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1) # B, N, C, L, 1

        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)

        output = self.input_embedding(long_term_history)

        output = self.norm_layer(output)

        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)    # B, N, d, P
        
        assert output.shape[-1] == len_time_series / self.len_patch
        return output
