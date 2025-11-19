import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import Board

import logging

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class GomokuNet(nn.Module):
    def __init__(self, board_width=15, board_height=15, num_res_blocks=4, num_filters=64):
        super(GomokuNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters

        # Input: 4 channels (Player stones, Opponent stones, Last move, Color to play)
        self.conv_input = nn.Conv2d(4, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 4, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch_size, 4, board_width, board_height)
        out = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(-1, 4 * self.board_width * self.board_height)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 2 * self.board_width * self.board_height)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class PolicyValueNet:
    """
    The policy-value network wrapper
    """
    def __init__(self, board_width=15, board_height=15, model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"PolicyValueNet initialized on GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("PolicyValueNet initialized on CPU")

        self.policy_value_net = GomokuNet(board_width, board_height).to(self.device)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    torch.from_numpy(current_state).float().to(self.device))
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                    torch.from_numpy(current_state).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
            
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = torch.FloatTensor(np.array(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
            mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)
            winner_batch = torch.FloatTensor(np.array(winner_batch)).to(self.device)
        else:
            state_batch = torch.FloatTensor(np.array(state_batch))
            mcts_probs = torch.FloatTensor(np.array(mcts_probs))
            winner_batch = torch.FloatTensor(np.array(winner_batch))

        # zero the parameter gradients
        optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=lr, weight_decay=self.l2_const)
        optimizer.zero_grad()

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the value head output is tanh, so it's in [-1, 1]
        # winner_batch is also in [-1, 1]
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        
        # backward and optimize
        loss.backward()
        optimizer.step()
        
        # entropy for monitoring
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
