from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import random, time
from typing import Dict, Tuple, List

from hex_player_helper import HexPlayerHelper, MCTSNode
from hex_player_debug import HexPlayerDebug


class MyPlayer(PlayerHex):
    
    W_CENTER_BASE = 5.0
    W_CENTER_DECAY = 0.1
    W_PROGRESS = 2.0
    W_BRIDGE = 4.0
    W_BLOCK = 8.0
    W_PATH = 10.0
    W_BRIDGE_DEFENSE = 100.0
    
    W_VITAL_POINT = 6.0
    W_CORRIDOR = 1.5
    W_CONNECTIVITY = 3.0
    W_LADDER_BREAK = 7.0
    OVERCONCENTRATION_PENALTY = 0.6
    
    BONUS_CUT = 8.0
    BONUS_BRIDGE = 4.0
    
    BRIDGE_PATTERNS = [
        ((0, 0), (1, 1), (0, 1), (1, 0)),
        ((0, 0), (-1, -1), (0, -1), (-1, 0)),
        ((0, 0), (0, 2), (0, 1), (-1, 1)),
        ((0, 0), (0, 2), (0, 1), (1, 1)),
        ((0, 0), (1, 1), (0, 1), (1, 0)),
        ((0, 0), (-1, -1), (-1, 0), (0, -1)),
        ((0, 0), (2, 0), (1, 0), (1, -1)),
        ((0, 0), (2, -1), (1, 0), (1, -1)),
    ]

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self.opp_type = "B" if piece_type == "R" else "R"
        
        self._bridge_offsets = self._precompute_bridge_offsets()
        self._helper = HexPlayerHelper(self)
        self._debug = HexPlayerDebug(self, self._helper)
        
        goal_direction = "TOP→BOTTOM" if piece_type == "R" else "LEFT→RIGHT"
        print(f"🎯 Player initialized: {name} as {piece_type} (Goal: {goal_direction})")

    def _precompute_bridge_offsets(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        neighbors = [
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
        ]

        def neighbor_cells(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            i, j = pos
            return [(i + di, j + dj) for di, dj in neighbors]

        origin = (0, 0)
        origin_neighbors = set(neighbor_cells(origin))

        bridge_patterns = []

        for i, (d1i, d1j) in enumerate(neighbors):
            for j, (d2i, d2j) in enumerate(neighbors):
                if i >= j:
                    continue

                partner = (d1i + d2i, d1j + d2j)

                if partner == origin or partner in origin_neighbors:
                    continue

                partner_neighbors = set(neighbor_cells(partner))

                common = origin_neighbors & partner_neighbors

                if len(common) != 2:
                    continue

                carrier1 = (d1i, d1j)
                carrier2 = (d2i, d2j)

                if {carrier1, carrier2} != common:
                    continue

                bridge_patterns.append((partner, carrier1, carrier2))

        return bridge_patterns

    def compute_per_move_budget(
            self,
            remaining_time: int,
            fraction: float = 0.08,
            min_ms: int = 150,
            max_ms: int = 15000
    ) -> int:
        if remaining_time < 10000:
            rem_ms = remaining_time * 1000
        else:
            rem_ms = remaining_time

        budget = int(rem_ms * fraction)

        budget = max(min_ms, min(budget, max_ms, rem_ms - 50))

        return max(50, budget)

    def compute_action(
            self,
            current_state: GameState,
            remaining_time: int = int(1e9),
            **kwargs
    ) -> Action:
        if kwargs.get("rng_seed") is not None:
            random.seed(kwargs["rng_seed"])

        if current_state.is_done():
            actions = list(current_state.get_possible_heavy_actions())
            if not actions:
                raise MethodNotImplementedError("No legal actions in terminal state")
            return current_state.convert_heavy_action_to_light_action(actions[0])
        
        self._debug.print_bridge_analysis(current_state)
        
        actions = list(current_state.get_possible_heavy_actions())
        if not actions:
            raise MethodNotImplementedError("No legal actions available")
        
        if len(actions) == 1:
            return current_state.convert_heavy_action_to_light_action(actions[0])
        
        budget_ms = self.compute_per_move_budget(
            remaining_time,
            fraction=float(kwargs.get("per_move_fraction", 0.08)),
            min_ms=int(kwargs.get("per_move_min_ms", 150)),
            max_ms=int(kwargs.get("per_move_max_ms", 15000))
        )
        deadline = time.time() + budget_ms / 1000.0
        
        root = MCTSNode(current_state, parent=None, action=None, player_ref=self)
        
        iterations = 0
        while time.time() < deadline:
            node = self._helper.mcts_select(root)
            
            if not node.is_terminal and not node.is_fully_expanded():
                node = self._helper.mcts_expand(node)
            
            value = self._helper.mcts_simulate(node)
            
            self._helper.mcts_backpropagate(node, value)
            
            iterations += 1
        
        if not root.children:
            ordered_actions = self._helper.ordered_root_actions(current_state)
            return current_state.convert_heavy_action_to_light_action(ordered_actions[0])
        
        best_action, best_child = root.most_visited_child()
        
        print(f"🌲 Tree MCTS: {iterations} iterations, {len(root.children)} root children")
        print(f"   Best move visits: {best_child.visits}, avg value: {best_child.total_value/max(1, best_child.visits):.2f}")
        
        max_depth = self._helper._get_tree_depth(root)
        print(f"   Tree depth reached: {max_depth}")
        
        return current_state.convert_heavy_action_to_light_action(best_action)

    @classmethod
    def create_with_weights(cls, piece_type: str, name: str = "TunedPlayer",
                           w_center_base: float = 5.0, w_center_decay: float = 0.3,
                           w_progress: float = 2.0, w_bridge: float = 4.0,
                           w_block: float = 8.0, w_path: float = 10.0,
                           w_bridge_defense: float = 100.0,
                           w_vital_point: float = 6.0, w_corridor: float = 1.5,
                           w_connectivity: float = 3.0, w_ladder_break: float = 7.0,
                           overconcentration_penalty: float = 0.6,
                           bonus_cut: float = 8.0, bonus_bridge: float = 4.0) -> 'MyPlayer':
        player = cls(piece_type, name)
        player.W_CENTER_BASE = w_center_base
        player.W_CENTER_DECAY = w_center_decay
        player.W_PROGRESS = w_progress
        player.W_BRIDGE = w_bridge
        player.W_BLOCK = w_block
        player.W_PATH = w_path
        player.W_BRIDGE_DEFENSE = w_bridge_defense
        player.W_VITAL_POINT = w_vital_point
        player.W_CORRIDOR = w_corridor
        player.W_CONNECTIVITY = w_connectivity
        player.W_LADDER_BREAK = w_ladder_break
        player.OVERCONCENTRATION_PENALTY = overconcentration_penalty
        player.BONUS_CUT = bonus_cut
        player.BONUS_BRIDGE = bonus_bridge
        return player

    def get_weights(self) -> Dict[str, float]:
        return {
            'W_CENTER_BASE': self.W_CENTER_BASE,
            'W_CENTER_DECAY': self.W_CENTER_DECAY,
            'W_PROGRESS': self.W_PROGRESS,
            'W_BRIDGE': self.W_BRIDGE,
            'W_BLOCK': self.W_BLOCK,
            'W_PATH': self.W_PATH,
            'W_BRIDGE_DEFENSE': self.W_BRIDGE_DEFENSE,
            'W_VITAL_POINT': self.W_VITAL_POINT,
            'W_CORRIDOR': self.W_CORRIDOR,
            'W_CONNECTIVITY': self.W_CONNECTIVITY,
            'W_LADDER_BREAK': self.W_LADDER_BREAK,
            'OVERCONCENTRATION_PENALTY': self.OVERCONCENTRATION_PENALTY,
            'BONUS_CUT': self.BONUS_CUT,
            'BONUS_BRIDGE': self.BONUS_BRIDGE,
        }

    def set_weights(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
