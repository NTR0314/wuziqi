from game import Board, Game
from mcts import MCTSPlayer
from model import PolicyValueNet
import numpy as np

def run():
    n = 5
    width, height = 8, 8 # Smaller board for faster test
    model_file = None
    
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        
        # Use a fresh model
        policy_value_net = PolicyValueNet(width, height, model_file)
        
        # Create MCTS player
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=50)
        
        # Play a few moves
        print("Starting self-play test...")
        winner, data = game.start_self_play(mcts_player, is_shown=0, temp=1e-3)
        print(f"Game finished. Winner: {winner}")
        print(f"Data collected: {len(list(data))} steps")
        
    except Exception as e:
        print(f"FAILED: {e}")
        raise e

if __name__ == '__main__':
    run()
