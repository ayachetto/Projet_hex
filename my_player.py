from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math

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

    def minimax(self, state: GameState, maximizing_player: bool):
        """
        Minimax recursive search.

        Args:
            state (GameState): Current state.
            maximizing_player (bool): True if it's this player's turn.

        Returns:
            (score, action): Best score and action.
        """
        # If terminal state, return final score difference
        if state.is_done():
            # score for self minus opponent
            my_score = state.get_player_score(self)
            opp_score = state.get_player_score(state.compute_next_player())
            return my_score - opp_score, None

        if maximizing_player:
            best_value = -math.inf
            best_action = None
            for action in state.generate_possible_heavy_actions():
                next_state = action.get_next_game_state()
                value, _ = self.minimax(next_state, False)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action
        else:
            best_value = math.inf
            best_action = None
            for action in state.generate_possible_heavy_actions():
                next_state = action.get_next_game_state()
                value, _ = self.minimax(next_state, True)
                if value < best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        # Run minimax assuming it's our turn (maximizing)
        _, best_action = self.minimax(current_state, True)
        if best_action is None:
            # fallback: pick any legal action if minimax fails
            actions = list(current_state.generate_possible_heavy_actions())
            return current_state.convert_heavy_action_to_light_action(actions[0])
        return current_state.convert_heavy_action_to_light_action(best_action)
        #TODO
        #raise MethodNotImplementedError()
