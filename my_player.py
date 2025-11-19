from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math, random, time
from typing import Dict, Tuple, List
import heapq


class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
        self.opp_type = "B" if piece_type == "R" else "R"

    # ========== FAST HEURISTIC (for rollouts) ==========

    def fast_heuristic(self, state: GameState) -> float:
        """
        Fast position evaluation for rollouts.
        Called thousands of times, must be VERY fast.

        Components:
        - Center control: pieces near center are better
        - Progress: pieces advancing toward goal are better
        """
        # Terminal state: return actual result
        if state.is_done():
            if state.get_player_score(self) > 0:
                return 1000.0  # We won
            else:
                return -1000.0  # We lost

        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0

        my_score = 0.0
        opp_score = 0.0

        for (i, j), piece in env.items():
            # Center bonus: pieces closer to center are more valuable
            dist_from_center = abs(i - center) + abs(j - center)
            center_value = 2.0 / (1.0 + dist_from_center * 0.1)

            # Progress bonus: advancing toward goal
            if piece.get_type() == self.piece_type:
                if self.piece_type == "R":
                    progress = i / (dim - 1)  # Advance from top (row 0) to bottom (row N-1)
                else:  # Blue
                    progress = j / (dim - 1)  # Advance from left (col 0) to right (col N-1)

                my_score += center_value + progress * 2.0

            else:  # Opponent's piece
                if self.opp_type == "R":
                    progress = i / (dim - 1)
                else:
                    progress = j / (dim - 1)

                opp_score += center_value + progress * 2.0

        return my_score - opp_score

    # ========== ACCURATE HEURISTIC (for root move ordering) ==========

    def accurate_heuristic(self, state: GameState) -> float:
        """
        More accurate but slower position evaluation.
        Used for ordering root moves (called ~100 times per turn).

        Uses shortest path distance calculation via Dijkstra's algorithm.
        """
        # Terminal state
        if state.is_done():
            if state.get_player_score(self) > 0:
                return 1000.0
            else:
                return -1000.0

        # Calculate shortest path distances for both players
        my_dist = self.shortest_path_distance(state, self.piece_type)
        opp_dist = self.shortest_path_distance(state, self.opp_type)

        # Being closer to goal is better
        # Weight heavily (10x) because path distance is critical
        return (opp_dist - my_dist) * 10.0

    def shortest_path_distance(self, state: GameState, piece_type: str) -> float:
        """
        Calculate minimum number of empty cells needed to connect sides.
        Uses Dijkstra's algorithm.

        Cost model:
        - Own pieces: 0 (already placed)
        - Empty cells: 1 (need to place a piece)
        - Opponent pieces: infinity (cannot use)
        """
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        visited = set()
        pq = []  # Priority queue: (distance, i, j)

        opp_type = "B" if piece_type == "R" else "R"

        # Initialize starting positions
        if piece_type == "R":
            # Red starts from top row (i=0, any j)
            for j in range(dim):
                if (0, j) in env and env[(0, j)].get_type() == piece_type:
                    cost = 0  # Already have a piece here
                elif (0, j) not in env:
                    cost = 1  # Empty, need to place
                else:
                    continue  # Opponent piece, skip
                heapq.heappush(pq, (cost, 0, j))
        else:  # Blue
            # Blue starts from left column (any i, j=0)
            for i in range(dim):
                if (i, 0) in env and env[(i, 0)].get_type() == piece_type:
                    cost = 0
                elif (i, 0) not in env:
                    cost = 1
                else:
                    continue
                heapq.heappush(pq, (cost, i, 0))

        # Dijkstra's algorithm
        while pq:
            dist, i, j = heapq.heappop(pq)

            if (i, j) in visited:
                continue
            visited.add((i, j))

            # Check if reached goal
            if piece_type == "R" and i == dim - 1:
                return dist  # Reached bottom row
            if piece_type == "B" and j == dim - 1:
                return dist  # Reached right column

            # Explore neighbors
            neighbors = state.get_neighbours(i, j)
            for neighbor_type, (ni, nj) in neighbors.values():
                if (ni, nj) in visited:
                    continue

                if neighbor_type == "OUTSIDE":
                    continue
                elif neighbor_type == piece_type:
                    # Our piece: free to use
                    new_dist = dist
                elif neighbor_type == "EMPTY":
                    # Empty: need to place a piece
                    new_dist = dist + 1
                else:
                    # Opponent piece: cannot use
                    continue

                heapq.heappush(pq, (new_dist, ni, nj))

        # No path found (shouldn't happen in valid Hex)
        return dim

    # ========== ROLLOUT POLICY ==========

    def biased_rollout(self, state: GameState) -> float:
        """
        Epsilon-greedy rollout using fast heuristic.
        Explores randomly sometimes, exploits heuristic other times.
        """
        s = state
        steps = 0
        max_depth = 100  # Prevent infinite loops

        while not s.is_done() and steps < max_depth:
            actions = list(s.get_possible_heavy_actions())
            if not actions:
                break

            # Epsilon-greedy: explore vs exploit
            # More exploration early in rollout, less later
            epsilon = 0.25 if steps < 5 else 0.15

            if random.random() < epsilon:
                # Explore: random action
                action = random.choice(actions)
            else:
                # Exploit: best action according to fast heuristic
                action = max(actions,
                             key=lambda a: self.fast_heuristic(a.get_next_game_state()))

            s = action.get_next_game_state()
            steps += 1

        # Return heuristic evaluation of final state
        return self.fast_heuristic(s)

    # ========== ROOT MOVE ORDERING ==========

    def ordered_root_actions(self, state: GameState) -> List:
        """
        Order root moves by quality using accurate heuristic.
        Best moves first helps Monte Carlo focus on promising branches.
        """
        actions = list(state.get_possible_heavy_actions())
        if not actions:
            return actions

        # Separate immediate wins from other moves
        wins = []
        others = []

        for action in actions:
            next_state = action.get_next_game_state()

            if next_state.is_done() and next_state.get_player_score(self) > 0:
                # Immediate win!
                wins.append((action, float('inf')))
            else:
                # Evaluate with accurate heuristic
                value = self.accurate_heuristic(next_state)
                others.append((action, value))

        # Sort by value (descending)
        others.sort(key=lambda x: -x[1])

        # Return wins first, then others
        return [action for action, _ in wins + others]

    # ========== UCB (Upper Confidence Bound) ==========

    @staticmethod
    def ucb(mean: float, visits: int, total_visits: int, c: float = 1.4) -> float:
        """
        UCB1 formula for action selection.
        Balances exploitation (high mean) with exploration (low visits).
        """
        if visits == 0:
            return float('inf')  # Always try untried actions first

        exploitation = mean
        exploration = c * math.sqrt(math.log(max(1, total_visits)) / visits)
        return exploitation + exploration

    def pick_ucb_arm(self, stats: Dict[int, Tuple[int, float]], total: int) -> int:
        """Select action index with highest UCB value."""
        best_idx, best_ucb = None, -1e18

        for idx, (visits, sum_rewards) in stats.items():
            mean = (sum_rewards / visits) if visits > 0 else 0.0
            ucb_value = self.ucb(mean, visits, total)

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_idx = idx

        return best_idx

    # ========== TIME MANAGEMENT ==========

    def compute_per_move_budget(
            self,
            remaining_time: int,
            fraction: float = 0.08,
            min_ms: int = 150,
            max_ms: int = 15000
    ) -> int:
        """
        Allocate time for this move based on remaining time.
        Conservative: use ~8% of remaining time per move.
        """
        # Auto-detect if remaining_time is in seconds or milliseconds
        if remaining_time < 10000:
            # Likely seconds, convert to milliseconds
            rem_ms = remaining_time * 1000
        else:
            rem_ms = remaining_time

        # Take a fraction of remaining time
        budget = int(rem_ms * fraction)

        # Clamp to reasonable range
        budget = max(min_ms, min(budget, max_ms, rem_ms - 50))

        return max(50, budget)  # At least 50ms

    # ========== MAIN ENTRY POINT ==========

    def compute_action(
            self,
            current_state: GameState,
            remaining_time: int = int(1e9),
            **kwargs
    ) -> Action:
        """
        Main decision function: Monte Carlo Tree Search (flat, multi-armed bandit style).

        Algorithm:
        1. Get all legal moves
        2. Order them by heuristic quality
        3. Allocate time for this move
        4. Run UCB-guided simulations:
           - Select move with highest UCB
           - Simulate from that move using biased rollout
           - Update statistics
        5. Return move with best average value
        """
        # Set random seed if provided (for reproducibility)
        if kwargs.get("rng_seed") is not None:
            random.seed(kwargs["rng_seed"])

        # Handle terminal state (shouldn't happen)
        if current_state.is_done():
            actions = list(current_state.get_possible_heavy_actions())
            if not actions:
                raise MethodNotImplementedError("No legal actions in terminal state")
            return current_state.convert_heavy_action_to_light_action(actions[0])

        # Get and order legal moves
        actions = self.ordered_root_actions(current_state)

        if not actions:
            raise MethodNotImplementedError("No legal actions available")

        # If only one move, just play it
        if len(actions) == 1:
            return current_state.convert_heavy_action_to_light_action(actions[0])

        # Compute time budget for this move
        budget_ms = self.compute_per_move_budget(
            remaining_time,
            fraction=float(kwargs.get("per_move_fraction", 0.08)),
            min_ms=int(kwargs.get("per_move_min_ms", 150)),
            max_ms=int(kwargs.get("per_move_max_ms", 15000))
        )
        deadline = time.time() + budget_ms / 1000.0

        # Get next states for all actions (cached for efficiency)
        next_states = [action.get_next_game_state() for action in actions]

        # Initialize statistics: {action_index: (visits, sum_of_rewards)}
        stats: Dict[int, Tuple[int, float]] = {i: (0, 0.0) for i in range(len(actions))}
        total_simulations = 0

        # Phase 1: Initialize each action with at least one rollout
        for i in range(len(actions)):
            if time.time() >= deadline:
                break

            reward = self.biased_rollout(next_states[i])
            visits, sum_rewards = stats[i]
            stats[i] = (visits + 1, sum_rewards + reward)
            total_simulations += 1

        # Phase 2: UCB-guided simulation allocation
        while time.time() < deadline:
            # Select action with highest UCB
            idx = self.pick_ucb_arm(stats, total_simulations)

            # Run one rollout from that action
            reward = self.biased_rollout(next_states[idx])

            # Update statistics
            visits, sum_rewards = stats[idx]
            stats[idx] = (visits + 1, sum_rewards + reward)
            total_simulations += 1

        # Select action with best average reward
        best_idx = None
        best_mean = -1e18

        for i, (visits, sum_rewards) in stats.items():
            if visits > 0:
                mean = sum_rewards / visits
                if mean > best_mean:
                    best_mean = mean
                    best_idx = i

        # Fallback to first action if something went wrong
        if best_idx is None:
            best_idx = 0

        return current_state.convert_heavy_action_to_light_action(actions[best_idx])

