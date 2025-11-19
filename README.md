# Gomoku AlphaZero RL

A Reinforcement Learning agent for Gomoku (Five in a Row) based on the AlphaZero methodology.

## Features
- **AlphaZero Algorithm:** Combines MCTS (Monte Carlo Tree Search) with a deep residual neural network.
- **Multiprocessing:** Uses a Model Server and multiple Self-Play Workers to scale data generation on multi-core CPUs.
- **WandB Integration:** Logs metrics, game replays (HTML), and model checkpoints (Artifacts) to Weights & Biases.
- **Optimized MCTS:** Features virtual loss, GPU batching, and parallel execution.

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

## Usage

### Training

To start training the agent:

```bash
uv run python main.py --mode train
```

**Arguments:**
- `--debug`: Run in debug mode with fewer workers and simulations for testing.

### WandB Authentication (Remote Servers)

If you are running this on a remote server (especially one you don't fully trust), **do not run `wandb login`**, as it saves credentials to a file on disk.

Instead, use an environment variable to authenticate for the single session:

1.  Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).
2.  Run the training command with the key:

```bash
WANDB_API_KEY=your_api_key_here uv run python main.py --mode train
```

This keeps your credentials in memory only for that process.

## Architecture

- **`main.py`**: Entry point.
- **`train.py`**: Training pipeline, Model Server, and Worker logic.
- **`mcts.py`**: Monte Carlo Tree Search implementation.
- **`model.py`**: PyTorch neural network definition.
- **`game.py`**: Gomoku game rules and board logic.
