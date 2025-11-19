import argparse
from train import TrainPipeline
from game import Board, Game
from mcts import MCTSPlayer
from model import PolicyValueNet

def run_training(debug=False):
    print("Starting training...")
    pipeline = TrainPipeline(debug=debug)
    pipeline.run()

def play_game(model_file=None):
    print("Starting game...")
    board = Board(width=15, height=15, n_in_row=5)
    game = Game(board)
    
    if model_file:
        policy_value_net = PolicyValueNet(15, 15, model_file=model_file)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)
    else:
        print("No model provided, using random MCTS.")
        # Just a dummy function for random play if needed, or handle otherwise
        # For now, let's assume we always want a model for the AI
        policy_value_net = PolicyValueNet(15, 15)
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)

    # Human player logic would go here, or simple AI vs AI
    # For simplicity, let's do AI vs AI self-play visualization
    game.start_self_play(mcts_player, is_shown=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gomoku AlphaZero')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'], help='Mode: train or play')
    parser.add_argument('--model', type=str, default=None, help='Path to model file for play mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (low resource, logging)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training(args.debug)
    else:
        play_game(args.model)
