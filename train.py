import logging
import time
import os
import torch
import random
import numpy as np
import multiprocessing as mp
from collections import deque
from game import Board, Game
from mcts import MCTSPlayer
from model import PolicyValueNet
import wandb

# Configure logging
def setup_logger(log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    # Remove existing handlers to avoid duplicates if called multiple times
    logger.handlers = []
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

class PerformanceMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.inference_times = []
        self.wait_times = []
        self.training_times = []
        self.batch_sizes = []
        self.start_time = time.time()
        
    def log_inference(self, duration, batch_size):
        self.inference_times.append(duration)
        self.batch_sizes.append(batch_size)
        
    def log_wait(self, duration):
        self.wait_times.append(duration)
        
    def log_training(self, duration):
        self.training_times.append(duration)
        
    def get_stats(self):
        stats = {}
        if self.inference_times:
            stats['avg_inference_ms'] = np.mean(self.inference_times) * 1000
            stats['max_inference_ms'] = np.max(self.inference_times) * 1000
        if self.wait_times:
            stats['avg_wait_ms'] = np.mean(self.wait_times) * 1000
            stats['total_wait_s'] = np.sum(self.wait_times)
        if self.training_times:
            stats['avg_train_s'] = np.mean(self.training_times)
        if self.batch_sizes:
            stats['avg_batch_size'] = np.mean(self.batch_sizes)
            
        stats['duration_s'] = time.time() - self.start_time
        return stats

def self_play_worker(worker_id, conn, config, model_file=None):
    """
    Worker process that plays games against itself.
    conn: Connection to the model server (Pipe)
    """
    # Setup worker logger to write to main train.log
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.setLevel(logging.INFO)
    handler = logging.FileHandler("train.log")
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    worker_logger.addHandler(handler)
    
    worker_logger.info(f"Worker {worker_id} started")
    
    # Initialize environment
    board = Board(width=config['board_width'],
                  height=config['board_height'],
                  n_in_row=config['n_in_row'])
    game = Game(board)
    
    # Track statistics
    request_count = 0
    
    # Define the remote policy function
    def policy_value_fn(board):
        nonlocal request_count
        request_count += 1
        # Send request to server
        conn.send((board.current_state(), board.availables))
        # Wait for response
        action_probs, value = conn.recv()
        return action_probs, value

    # Initialize MCTS player with remote policy
    mcts_player = MCTSPlayer(policy_value_fn,
                             c_puct=config['c_puct'],
                             n_playout=config['n_playout'],
                             is_selfplay=1,
                             use_parallel=False)
    
    game_num = 0
    while True:
        # Play a game
        game_start = time.time()
        request_count = 0
        
        winner, play_data, moves = game.start_self_play(mcts_player, temp=config['temp'])
        
        game_duration = time.time() - game_start
        game_num += 1
        
        worker_logger.info(f"Worker {worker_id} Game {game_num}: {game_duration:.2f}s, "
                          f"{len(moves)} moves, {request_count} NN requests, winner: {winner}")
        
        # Send game data to server
        conn.send(("DATA", (winner, list(play_data), moves)))
        
        # No response expected for data

def evaluation_worker(worker_id, conn, config, best_model_file, current_model_file):
    """
    Worker for evaluating current model against best model.
    This is a bit complex because we need TWO models.
    Simplification: The server holds the 'current' model.
    We can have the server handle inference for 'current' player.
    But for 'best' player, we might need another model instance or server?
    
    Alternative: Just load the models here on CPU? Or GPU if available?
    If we have 1 GPU, we can't easily share it across processes without a server.
    
    Let's stick to the plan: "Play N games between them."
    If we want to use the GPU server for both, we need to distinguish requests.
    
    For now, to keep it simple, let's load the models on CPU in this worker. 
    Evaluation is less frequent, so CPU inference might be acceptable.
    Or, if we have enough VRAM, we can load them on GPU here too? 
    But CUDA context sharing is tricky.
    
    Let's try CPU inference for evaluation to avoid complexity.
    """
    try:
        board = Board(width=config['board_width'],
                      height=config['board_height'],
                      n_in_row=config['n_in_row'])
        game = Game(board)
        
        # Load policies
        # We need to handle the case where files might not exist yet or are being written
        time.sleep(1) # Wait a bit for file sync
        
        if not os.path.exists(best_model_file) or not os.path.exists(current_model_file):
            return 0.0
            
        policy_best = PolicyValueNet(config['board_width'], config['board_height'], model_file=best_model_file, use_gpu=False)
        policy_curr = PolicyValueNet(config['board_width'], config['board_height'], model_file=current_model_file, use_gpu=False)
        
        mcts_best = MCTSPlayer(policy_best.policy_value_fn, c_puct=config['c_puct'], n_playout=config['n_playout'])
        mcts_curr = MCTSPlayer(policy_curr.policy_value_fn, c_puct=config['c_puct'], n_playout=config['n_playout'])
        
        win_cnt = 0
        n_games = 10
        for i in range(n_games):
            # start_player=0 -> current goes first
            # start_player=1 -> best goes first
            # We want to alternate
            winner = game.start_play(mcts_curr, mcts_best, start_player=i % 2, is_shown=0)
            if winner == mcts_curr.player:
                win_cnt += 1
        
        return win_cnt / n_games
    except Exception as e:
        print(f"Eval error: {e}")
        return 0.0

class TrainPipeline:
    def __init__(self, init_model=None, debug=False):
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.debug = debug
        
        # Logging
        self.logger = setup_logger("train.log")
        self.monitor = PerformanceMonitor()
        
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        
        # Number of self-play workers based on CPU count
        self.num_workers = mp.cpu_count()
        if self.debug:
            self.num_workers = 2
            self.n_playout = 50
            self.batch_size = 2
            self.buffer_size = 100
            self.epochs = 1
            self.check_freq = 2
            self.game_batch_num = 10
            
        self.config = {
            "board_width": self.board_width,
            "board_height": self.board_height,
            "n_in_row": self.n_in_row,
            "n_playout": self.n_playout,
            "c_puct": self.c_puct,
            "temp": self.temp
        }

        # Initialize model
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
            
        # Save initial models
        self.policy_value_net.save_model('./current_policy.model')
        self.policy_value_net.save_model('./best_policy.model')

        # WandB
        wandb.init(project="gomoku-rl", config=self.config)
        
        self.logger.info(f"Training started with config: {self.config}")

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping"""
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def policy_update(self):
        """update the policy-value net"""
        start_time = time.time()
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1))
            if kl > self.kl_targ * 4:
                break
        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        wandb.log({
            "loss": loss,
            "entropy": entropy,
            "kl": kl,
            "lr_multiplier": self.lr_multiplier,
            "learning_rate": self.learn_rate * self.lr_multiplier
        })
        
        self.monitor.log_training(time.time() - start_time)
        return loss, entropy

    def run(self):
        # Start workers
        workers = []
        pipes = []
        
        for i in range(self.num_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=self_play_worker, args=(i, child_conn, self.config))
            p.start()
            workers.append(p)
            pipes.append(parent_conn)
            
        self.logger.info(f"Started {self.num_workers} workers (CPU count: {mp.cpu_count()})")
        
        try:
            game_count = 0
            while game_count < self.game_batch_num:
                # Model Server Loop
                # Collect requests
                wait_start = time.time()
                ready_pipes = mp.connection.wait(pipes, timeout=0.01)
                self.monitor.log_wait(time.time() - wait_start)
                
                requests = []
                request_pipes = []
                request_availables = []
                
                for pipe in ready_pipes:
                    try:
                        msg = pipe.recv()
                        # Check if msg is data using strict type checking to avoid numpy ambiguity
                        if isinstance(msg, tuple) and isinstance(msg[0], str) and msg[0] == "DATA":
                            # Handle game data
                            _, (winner, play_data, moves) = msg
                            play_data = self.get_equi_data(play_data)
                            self.data_buffer.extend(play_data)
                            game_count += 1
                            self.logger.info(f"Game {game_count} collected. Buffer size: {len(self.data_buffer)}")
                            
                            # Log sample game
                            if game_count % 10 == 0:
                                from html_logger import HtmlLogger
                                logger = HtmlLogger()
                                logger.save_game(moves, winner, filename=f"game_{game_count}.html")
                                wandb.log({"game_replay": wandb.Html(open(f"logs/game_{game_count}.html"))})

                            # Check for training
                            if len(self.data_buffer) > self.batch_size:
                                self.policy_update()
                                
                            # Check for evaluation
                            if game_count % self.check_freq == 0:
                                self.policy_value_net.save_model('./current_policy.model')
                                self.logger.info("Evaluating...")
                                win_ratio = evaluation_worker(0, None, self.config, './best_policy.model', './current_policy.model')
                                self.logger.info(f"Win ratio: {win_ratio}")
                                wandb.log({"win_ratio": win_ratio})
                                
                                if win_ratio > 0.55: # Slight bias towards challenger
                                    self.logger.info("New best policy!")
                                    self.best_win_ratio = win_ratio
                                    self.policy_value_net.save_model('./best_policy.model')
                                    
                                    # Log model artifact to WandB
                                    artifact = wandb.Artifact('gomoku-policy', type='model')
                                    artifact.add_file('./best_policy.model')
                                    wandb.log_artifact(artifact)
                                
                        else:
                            # Prediction request: (state, availables)
                            state, availables = msg
                            requests.append(state)
                            request_availables.append(availables)
                            request_pipes.append(pipe)
                            self.logger.info(f"Requests growing: current size {len(requests)}")
                    except EOFError:
                        pass
                
                # Batch inference
                if requests:
                    inference_start = time.time()
                    state_batch = np.array(requests)
                    self.logger.info(f"Performing inference on device: {next(self.policy_value_net.policy_value_net.parameters()).device}")
                    act_probs, values = self.policy_value_net.policy_value(state_batch)
                    self.monitor.log_inference(time.time() - inference_start, len(requests))
                    
                    for i, pipe in enumerate(request_pipes):
                        # Filter legal moves
                        legal_moves = request_availables[i]
                        probs = act_probs[i]
                        # We need to return a list of (action, prob) tuples
                        legal_probs = list(zip(legal_moves, probs[legal_moves]))
                        
                        pipe.send((legal_probs, values[i]))
                
                # Log stats periodically
                if game_count > 0 and game_count % 100 == 0:
                    stats = self.monitor.get_stats()
                    self.logger.info(f"Stats at game {game_count}: {stats}")
                    wandb.log(stats)
                    self.monitor.reset()
                
        except KeyboardInterrupt:
            self.logger.info("Stopping...")
        finally:
            # Upload log file to WandB
            if os.path.exists("train.log"):
                wandb.save("train.log")
                self.logger.info("Uploaded train.log to WandB")
            
            for p in workers:
                p.terminate()
                p.join()

if __name__ == '__main__':
    # Set start method to spawn for CUDA compatibility if needed, though fork is default on Linux
    # mp.set_start_method('spawn') 
    training_pipeline = TrainPipeline()
    training_pipeline.run()
