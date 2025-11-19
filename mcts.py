import numpy as np
import copy
import torch
import threading
from concurrent.futures import ThreadPoolExecutor

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # a map from action to TreeNode
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.P = prior_p
        self.virtual_loss = 0
        self.lock = threading.Lock()

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
        according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
        value Q, and prior probability P, on this node's score.
        """
        self.u = (c_puct * self.P *
                  np.sqrt(self.parent.n_visits + self.parent.virtual_loss) / (1 + self.n_visits + self.virtual_loss))
        return self.Q - self.virtual_loss + self.u

    def apply_virtual_loss(self):
        with self.lock:
            self.virtual_loss += 1

    def remove_virtual_loss(self):
        with self.lock:
            self.virtual_loss -= 1

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
        perspective.
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS:
    """An implementation of Monte Carlo Tree Search."""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a parameter controlling the level of exploration.
        n_playout: number of simulations for each search.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def get_move_probs(self, board, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # Use parallel playouts if configured (we'll default to sequential if not)
        # But here we will just run sequential loop for now, or we can switch to parallel
        # if we want to enforce it.
        # For now, let's keep this sequential and add a parallel version.
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(board)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def get_move_probs_parallel(self, board, temp=1e-3, num_threads=8):
        """Run playouts in parallel."""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(self._n_playout):
                state_copy = copy.deepcopy(board)
                futures.append(executor.submit(self._playout_parallel, state_copy))
            
            # Wait for all to complete
            for f in futures:
                f.result()

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.n_visits)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root.parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def _playout(self, board):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            board.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples and a score for the current player.
        action_probs, leaf_value = self._policy(board)

        # Check for end of game.
        end, winner = board.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == board.get_current_player() else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _playout_parallel(self, board):
        """Run a single playout in parallel with virtual loss."""
        node = self._root
        visited_nodes = []
        
        # Selection
        while True:
            with node.lock:
                if node.is_leaf():
                    break
                # Greedily select next move.
                action, node = node.select(self._c_puct)
                node.apply_virtual_loss()
                visited_nodes.append(node)
            
            board.do_move(action)

        # Evaluation
        # This policy call should be blocking and thread-safe (handled by batching worker)
        action_probs, leaf_value = self._policy(board)

        # Check for end of game.
        end, winner = board.game_end()
        if not end:
            with node.lock:
                # Double check if it's still a leaf (another thread might have expanded it)
                if node.is_leaf():
                    node.expand(action_probs)
                else:
                    # If already expanded, we might want to continue selection? 
                    # For simplicity, just backpropagate the value we got.
                    pass
        else:
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == board.get_current_player() else -1.0

        # Backpropagation
        # Remove virtual loss and update
        for n in visited_nodes:
            n.remove_virtual_loss()
        
        node.update_recursive(-leaf_value)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0, use_parallel=True):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.use_parallel = use_parallel
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            if self.use_parallel:
                acts, probs = self.mcts.get_move_probs_parallel(board, temp)
            else:
                acts, probs = self.mcts.get_move_probs(board, temp)
            
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                # location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
