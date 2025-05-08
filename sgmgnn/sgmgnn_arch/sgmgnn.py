import torch
from torch import nn
from .llm_encoder import LLMEncoder
from .llmencoder import LLMEncoder
from .graphwavenet import GraphWaveNet
from .discrete_graph_learning import DiscreteGraphLearning


class SGMGNN(nn.Module):
    

    # Define a class SGMGNN that inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
    def __init__(self, dataset_name, pre_trained_llmencoder_path, llmencoder_args, backend_args, dgl_args):
        # Constructor method to initialize the SGMGNN class.
        super().__init__()
        # Call the constructor of the parent class nn.Module to ensure proper initialization.
        self.dataset_name = dataset_name
        # Store the dataset name as an instance variable.
        self.pre_trained_llmencoder_path = pre_trained_llmencoder_path


        # Store the path to the pre-trained LLMEncoder model as an instance variable.
        self.llmencoder = LLMEncoder(**llmencoder_args)
        # Initialize the LLMEncoder with the provided arguments and store it as an instance variable.
        self.backend = GraphWaveNet(**backend_args)


        # Initialize the GraphWaveNet with the provided arguments and store it as an instance variable.
        self.load_pre_trained_model()


        # Call the method to load the pre-trained model weights into the LLMEncoder.
        self.discrete_graph_learning = DiscreteGraphLearning(**dgl_args)

        # Initialize the DiscreteGraphLearning module with the provided arguments and store it as an instance variable.
    def load_pre_trained_model(self):
        


        # Method to load the pre-trained model weights into the LLMEncoder.
        checkpoint_dict = torch.load(self.pre_trained_llmencoder_path)
        # Load the checkpoint dictionary from the specified file path.
        self.llmencoder.load_state_dict(checkpoint_dict["model_state_dict"])

        # Load the state dictionary (model weights) into the LLMEncoder.
        for param in self.llmencoder.parameters():
            param.requires_grad = False

        # Set all parameters of the LLMEncoder to not require gradients, effectively freezing the model.
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        


        # Define the forward pass of the model.
        short_term_history = history_data     # [B, L, N, 1]
        # Assign the short-term history data to a variable. The shape is [Batch Size, Sequence Length, Number of Nodes, 1].
        long_term_history = long_history_data


        # Assign the long-term history data to a variable.
        batch_size, _, num_nodes, _ = short_term_history.shape


        # Extract the batch size, sequence length, number of nodes, and feature dimension from the short-term history data.
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.discrete_graph_learning(long_term_history, self.llmencoder)


        # Call the discrete_graph_learning method to get the unnormalized bernoulli distribution, hidden states, k-NN adjacency matrix, and sampled adjacency matrix.
        hidden_states = hidden_states[:, :, -1, :]
        # Select the last hidden state from the sequence for each node.
        y_hat = self.backend(short_term_history, hidden_states=hidden_states, sampled_adj=sampled_adj).transpose(1, 2)


        # Pass the short-term history, last hidden states, and sampled adjacency matrix through the backend model and transpose the output.
        if epoch is not None:
            gsl_coefficient = 1 / (int(epoch/6)+1)
        else:
            gsl_coefficient = 0
        # Calculate the graph sampling and learning coefficient based on the current epoch. If epoch is None, set the coefficient to 0.
        return y_hat.unsqueeze(-1), bernoulli_unnorm.softmax(-1)[..., 0].clone().reshape(batch_size, num_nodes, num_nodes), adj_knn, gsl_coefficient
