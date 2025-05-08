import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from torchsummary import summary
from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from .StandardNorm import Normalize
from transformers import  BertConfig, BertModel, BertTokenizer
from math import sqrt
def unshuffle(shuffled_tokens):
    # Initialize an empty dictionary to store the original index of each token
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index
class LLMEncoder(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth, mode="pre-train"):
        super().__init__()  
        # Ensure the mode is either "pre-train" or "forecasting"
        assert mode in ["pre-train", "forecasting"], "Error mode."  
        # Initialize class attributes
        self.patch_size = patch_size  
        self.in_channel = in_channel  
        self.embed_dim = embed_dim  
        self.num_heads = num_heads  
        self.num_token = int(num_token)  
        self.mask_ratio = mask_ratio  
        self.mode = mode  
        self.mlp_ratio = mlp_ratio  
        self.llm_dim = 768  
        self.selected_feature = 0  
        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)  
        # Create mask generator
        self.mask = MaskGenerator(num_token, mask_ratio)  
        # Create output layer
        self.output_layer = nn.Linear(self.llm_dim, patch_size)  
        # Load BERT configuration and model
        self.bert_config = BertConfig.from_pretrained('sgmgnn/sgmgnn_arch/llm_encoder/bert-base-uncased')  
        self.bert_config.num_hidden_layers = 4  
        self.bert_config.output_attentions = True  
        self.bert_config.output_hidden_states = True  
        self.llm_model = BertModel.from_pretrained('sgmgnn/sgmgnn_arch/llm_encoder/bert-base-uncased')  
        # Load BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('sgmgnn/sgmgnn_arch/llm_encoder/bert-base-uncased')  
        # Set padding token if not already set
        if self.tokenizer.eos_token: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})  
            self.tokenizer.pad_token = pad_token
        # Freeze BERT model parameters
        for param in self.llm_model.parameters(): 
            param.requires_grad = False
        # Set top-k value for lag calculation
        self.top_k = 5        
        # Create reprogramming layer
        self.reprogramming_layer = ReprogrammingLayer(embed_dim, num_heads, d_llm = self.llm_dim, attention_dropout=0.1)  
        # Get word embeddings from BERT model
        self.word_embeddings = self.llm_model.get_input_embeddings().weight 
        self.vocab_size = self.word_embeddings.shape[0] 
        # Set number of tokens
        self.num_tokens = 1000
        # Create mapping layer
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens) 
        # Set dataset description
        self.description = 'The erythromycin production processes (EPP) dataset is collected by Dongyangguang biopharmaceutical company, which contains 14 sets of process data from different operation conditions.'
        # Create normalization layer
        self.normalize_layers = Normalize(14, affine=False)
    def encoding(self, long_term_history, mask=True):
        # Get batch size, sequence length, and feature dimensions
        B, N, _, T = long_term_history.shape  
        # Reshape and permute long-term history for prompt generation
        long_term_history_prompt = long_term_history.permute(0, 1, 3, 2).contiguous().reshape(B * N, T, 1) 
        # Calculate min, max, and median values
        min_values = torch.min(long_term_history_prompt, dim=1)[0] 
        max_values = torch.max(long_term_history_prompt, dim=1)[0]
        medians = torch.median(long_term_history_prompt, dim=1).values
        # Calculate lags and trends
        lags = self.calcute_lags(long_term_history_prompt) 
        trends = long_term_history_prompt.diff(dim=1).sum(dim=1)
        # Generate prompts
        prompt = []
        for b in range(long_term_history_prompt.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next 12 steps given the previous 2016 steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 12 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)
        # Tokenize prompts
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # Get prompt embeddings
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(long_term_history_prompt.device))
        prompt_embeddings = prompt_embeddings.reshape(B, N, -1, self.llm_dim).contiguous()
        # Get patch embeddings
        patches = self.patch_embedding(long_term_history)     
        patches = patches.transpose(-1, -2)         
        # Apply masking if required
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches
        # Reshape encoder input
        hidden_states_unmasked = encoder_input.view(B, N, -1, self.embed_dim)
        return hidden_states_unmasked, prompt_embeddings, unmasked_token_index, masked_token_index
    def calcute_lags(self, x_enc):
        # Calculate lags using FFT
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1) 
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1) 
        res = q_fft * torch.conj(k_fft) 
        corr = torch.fft.irfft(res, dim=-1) 
        mean_value = torch.mean(corr, dim=1) 
        _, lags = torch.topk(mean_value, self.top_k, dim=-1) 
        return lags
    def decoding(self, hidden_states_unmasked, prompt_embeddings, masked_token_index):
        # Get batch size and sequence length
        B, N, _, _ = hidden_states_unmasked.shape
        # Get source embeddings
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0) 
        # Reprogram hidden states
        hidden_states_unmasked = self.reprogramming_layer(hidden_states_unmasked, source_embeddings, source_embeddings) 
        # Concatenate prompt embeddings and hidden states
        hidden_states_full = torch.cat([prompt_embeddings, hidden_states_unmasked], dim=-2)   
        # Select relevant tokens
        hidden_states_full = hidden_states_full[:, :, -self.num_token:, :]
        assert hidden_states_full.shape == (B, N, self.num_token, self.llm_dim)
        # Reshape hidden states
        hidden_states_full = hidden_states_full.view(B*N, self.num_token, self.llm_dim) 
        # Decode using BERT model
        reconstruction_full = self.llm_model(inputs_embeds=hidden_states_full).last_hidden_state 
        # Reshape hidden states
        hidden_states_full = hidden_states_full.view(B, N, self.num_token, self.llm_dim)
        # Get full reconstruction
        reconstruction_full = self.output_layer(hidden_states_full) 
        return reconstruction_full
    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        # Get batch size and sequence length
        B, N, _, _ = reconstruction_full.shape
        # Get reconstructed masked tokens
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(B, N, -1).transpose(1, 2)     
        # Get real masked tokens
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() 
        label_masked_tokens = label_masked_tokens.view(B, -1, N)
        return reconstruction_masked_tokens, label_masked_tokens
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # Permute history data
        history_data = history_data.permute(0, 2, 3, 1)     
        B, N, _, _ = history_data.shape
        # Normalize history data
        history_data = self.normalize_layers(history_data, 'norm')
        # Pre-training mode
        if self.mode == "pre-train":
            hidden_states_unmasked, prompt_embeddings, unmasked_token_index, masked_token_index = self.encoding(history_data)
            reconstruction_full = self.decoding(hidden_states_unmasked, prompt_embeddings, masked_token_index)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index) 
            reconstruction_masked_tokens = self.normalize_layers(reconstruction_masked_tokens.reshape(B, N, 1, -1), 'denorm')
            reconstruction_masked_tokens = reconstruction_masked_tokens.reshape(B, -1, N)
            label_masked_tokens = self.normalize_layers(label_masked_tokens.reshape(B, N, 1, -1), 'denorm')
            label_masked_tokens = label_masked_tokens.reshape(B, -1, N)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            # Forecasting mode
            hidden_states_full, prompt_embeddings, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full, prompt_embeddings
class ReprogrammingLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_llm, attention_dropout=0.1):
        # Initialize the ReprogrammingLayer with the given dimensions and dropout rate
        super(ReprogrammingLayer, self).__init__()
        # Calculate the dimension of keys for each head
        d_keys = (embed_dim // num_heads)
        # Linear layer to project target embeddings to query space
        self.query_projection = nn.Linear(embed_dim, d_keys * num_heads)
        # Linear layer to project source embeddings to key space
        self.key_projection = nn.Linear(d_llm, d_keys * num_heads)
        # Linear layer to project value embeddings to value space
        self.value_projection = nn.Linear(d_llm, d_keys * num_heads)
        # Linear layer to project the output back to the original dimension
        self.out_projection = nn.Linear(d_keys * num_heads, d_llm)
        # Store the number of heads
        self.num_heads = num_heads
        # Dropout layer for regularization
        self.dropout = nn.Dropout(attention_dropout)
    def forward(self, target_embedding, source_embedding, value_embedding):
        # Get the batch size, sequence length, and embedding dimension from target embeddings
        B, N, L, _ = target_embedding.shape  
        # Get the sequence length of source embeddings
        S, _ = source_embedding.shape
        # Number of heads
        H = self.num_heads
        # Project target embeddings to query space and reshape for multi-head attention
        target_embedding = self.query_projection(target_embedding.view(-1, target_embedding.size(-1))) \
            .view(B, N, L, H, -1)
        # Project source embeddings to key space and reshape for multi-head attention
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        # Project value embeddings to value space and reshape for multi-head attention
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        # Perform the reprogramming operation
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        # Reshape the output to match the original target embedding shape
        out = out.reshape(B, N, L, -1)  
        # Project the output back to the original dimension
        return self.out_projection(out)
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        # Get the batch size, sequence length, number of heads, and embedding dimension
        B, N, L, H, E = target_embedding.shape
        # Scaling factor for the dot product
        scale = 1. / sqrt(E)
        # Compute the attention scores using Einstein summation convention
        scores = torch.einsum("bNlhe,she->bNhls", target_embedding, source_embedding)  
        # Apply softmax to the scores and apply dropout for regularization
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # Compute the weighted sum of value embeddings using the attention scores
        reprogramming_embedding = torch.einsum("bNhls,she->bNlhe", A, value_embedding)  
        # Return the reprogrammed embedding
        return reprogramming_embedding
