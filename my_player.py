from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math, random, time
from typing import Dict, Tuple, List, Set, Optional
import heapq


class MCTSNode:
    """
    Node in the MCTS tree for deeper lookahead.
    
    Each node represents a game state and tracks:
    - State reference (lightweight hash for deduplication)
    - Parent node and children nodes
    - Action that led to this state
    - Visit count and total value (for UCB calculation)
    - Unexpanded actions (actions not yet explored)
    - Terminal flag
    """
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Action] = None, player_ref = None):
        """
        Initialize an MCTS node.
        
        Args:
            state: The game state this node represents
            parent: Parent node in the tree (None for root)
            action: Action that led from parent to this node
            player_ref: Reference to the player (for heuristic evaluation)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.player_ref = player_ref
        
        # MCTS statistics
        self.visits = 0
        self.total_value = 0.0
        
        # Children: action -> MCTSNode
        self.children = {}
        
        # Unexpanded actions (will be populated lazily)
        self._unexpanded_actions = None
        self._all_actions = None
        
        # Terminal flag
        self.is_terminal = state.is_done()
    
    def get_unexpanded_actions(self) -> List[Action]:
        """Get list of actions not yet expanded as children."""
        if self._unexpanded_actions is None:
            if self.is_terminal:
                self._unexpanded_actions = []
                self._all_actions = []
            else:
                self._all_actions = list(self.state.get_possible_heavy_actions())
                self._unexpanded_actions = self._all_actions.copy()
        return self._unexpanded_actions
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been expanded."""
        return len(self.get_unexpanded_actions()) == 0
    
    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        """
        Select best child using UCB1 formula.
        
        Args:
            c: Exploration constant (higher = more exploration)
        
        Returns:
            Child node with highest UCB value
        """
        best_value = -float('inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                ucb_value = float('inf')
            else:
                # UCB1 formula: exploitation + exploration
                exploitation = child.total_value / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        
        return best_child
    
    def most_visited_child(self) -> Tuple[Action, 'MCTSNode']:
        """
        Return the child with the most visits (robust child selection).
        
        Returns:
            Tuple of (action, child_node) with most visits
        """
        best_visits = -1
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def update(self, value: float):
        """
        Update node statistics after a simulation.
        
        Args:
            value: Simulation result value
        """
        self.visits += 1
        self.total_value += value
    
    def __repr__(self):
        return f"MCTSNode(visits={self.visits}, value={self.total_value:.2f}, children={len(self.children)})"


class MyPlayer(PlayerHex):
    """
    Player class for Hex game with Bridge-focused MCTS.

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player

    Bridge Strategy:
    ----------------
    In Hex, a "bridge" is a virtual connection between two stones that cannot be broken.
    The classic pattern:
        .  X       or      X  .
       X  .               .  X

    If both empty cells are available, this is an unbreakable connection because
    if opponent plays one, we immediately play the other.

    This implementation prioritizes:
    1. Building bridges from center outward toward goal edges
    2. Extending and securing existing bridge chains
    3. Blocking opponent bridges at critical cut points
    """

    # ========== BRIDGE PATTERN DEFINITIONS ==========
    # Each bridge pattern is defined as:
    # (stone1_offset, stone2_offset, carrier1_offset, carrier2_offset)
    # where offsets are (di, dj) from the reference point

    # The 6 canonical bridge patterns in hexagonal coordinates
    # Pattern: two stones with two shared empty neighbors forming virtual connection
    BRIDGE_PATTERNS = [
        # Pattern 1: Stone at (0,0), Stone at (1,1), Carriers at (0,1) and (1,0)
        ((0, 0), (1, 1), (0, 1), (1, 0)),
        # Pattern 2: Stone at (0,0), Stone at (-1,-1), Carriers at (0,-1) and (-1,0)
        ((0, 0), (-1, -1), (0, -1), (-1, 0)),
        # Pattern 3: Stone at (0,0), Stone at (1,-1), Carriers at (0,-1) and (1,-1+1)=(1,0)
        # Actually: (0,0)-(1,-1) with carriers at (0,-1) and (1,0) - wait that's wrong
        # Let me reconsider the hex neighbor structure
        # Neighbors of (i,j): top_right(i-1,j+1), top_left(i-1,j), bot_left(i+1,j-1),
        #                     bot_right(i+1,j), left(i,j-1), right(i,j+1)
        # Bridge: two stones share exactly two empty common neighbors

        # Pattern 3: Vertical bridge - (0,0) to (2,0) via (1,0) and (1,-1)
        # Actually this is not a bridge, stones must be 2 apart with 2 shared neighbors

        # Let me redefine based on hex geometry:
        # A bridge connects stones that are "2-away" in a hex pattern
        # Pattern types based on direction:

        # Horizontal-ish bridges (good for Blue connecting left-right)
        ((0, 0), (0, 2), (0, 1), (-1, 1)),   # right-right with top carrier
        ((0, 0), (0, 2), (0, 1), (1, 1)),    # right-right with bottom carrier - wait need to check

        # Diagonal bridges (important for both players)
        ((0, 0), (1, 1), (0, 1), (1, 0)),    # Classic down-right diagonal
        ((0, 0), (-1, -1), (-1, 0), (0, -1)), # Up-left diagonal (mirror)

        # Vertical-ish bridges (good for Red connecting top-bottom)
        ((0, 0), (2, 0), (1, 0), (1, -1)),   # Down-down
        ((0, 0), (2, -1), (1, 0), (1, -1)),  # Down-left diagonal
    ]

    # Heuristic weights (tunable)
    W_CENTER_BASE = 5.0  # Center control BASE weight (decays exponentially)
    W_CENTER_DECAY = 0.1 # Decay rate: weight = BASE * (DECAY ^ game_progress)
    W_PROGRESS = 2.0     # Goal progress weight
    W_BRIDGE = 4.0       # Bridge formation/maintenance weight
    W_BLOCK = 8.0        # Blocking opponent bridges weight (increased for balance)
    W_PATH = 10.0        # Shortest path weight (most important)
    W_BRIDGE_DEFENSE = 100.0  # MUST defend threatened bridges (highest priority)
    
    # Enhanced blocking weights
    W_VITAL_POINT = 6.0      # Vital point blocking weight
    W_CORRIDOR = 1.5         # Corridor factor multiplier
    W_CONNECTIVITY = 3.0     # Local connectivity weight
    W_LADDER_BREAK = 7.0     # Ladder break bonus
    OVERCONCENTRATION_PENALTY = 0.6  # Penalty for clustering (< 1.0 reduces score)
    
    # Move ordering bonuses (controls blocking vs bridging balance)
    BONUS_CUT = 8.0      # Bonus for cut point moves (blocking)
    BONUS_BRIDGE = 4.0   # Bonus for bridge-forming moves

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
        self.opp_type = "B" if piece_type == "R" else "R"

        # Precompute all valid bridge patterns for efficiency
        self._bridge_offsets = self._precompute_bridge_offsets()
        
        # Debug: Verify correct goal direction
        goal_direction = "TOP→BOTTOM" if piece_type == "R" else "LEFT→RIGHT"
        print(f"🎯 Player initialized: {name} as {piece_type} (Goal: {goal_direction})")

    # ========== DYNAMIC WEIGHT CALCULATION ==========

    def get_dynamic_center_weight(self, state: GameState) -> float:
        """
        Calculate center weight that decays exponentially as game progresses.

        Formula: W_CENTER = W_CENTER_BASE * (W_CENTER_DECAY ^ game_progress)

        Where game_progress = current_step / max_possible_steps

        This means:
        - At game start (progress=0): weight = W_CENTER_BASE * 1.0 = 5.0
        - At 25% through:             weight ≈ 5.0 * 0.74 = 3.7
        - At 50% through:             weight ≈ 5.0 * 0.55 = 2.75
        - At 75% through:             weight ≈ 5.0 * 0.40 = 2.0
        - At game end (progress=1):   weight = 5.0 * 0.3 = 1.5

        The decay is exponential: y = base * (decay_rate ^ x)
        """
        # Get game progress (0 = start, 1 = all cells filled)
        current_step = state.get_step()
        max_steps = state.max_step  # dim * dim

        # Calculate progress ratio (clamped to [0, 1])
        progress = min(1.0, max(0.0, current_step / max_steps))

        # Exponential decay: weight = base * (decay ^ progress)
        # At progress=0: weight = base * 1 = base
        # At progress=1: weight = base * decay
        dynamic_weight = self.W_CENTER_BASE * (self.W_CENTER_DECAY ** progress)

        return dynamic_weight

    def get_game_phase(self, state: GameState) -> str:
        """
        Determine current game phase for strategic adjustments.

        Returns: 'opening', 'midgame', or 'endgame'
        """
        current_step = state.get_step()
        max_steps = state.max_step
        progress = current_step / max_steps

        if progress < 0.15:
            return 'opening'
        elif progress < 0.50:
            return 'midgame'
        else:
            return 'endgame'

    # ========== BRIDGE PATTERN PRECOMPUTATION ==========

    def _precompute_bridge_offsets(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """
        Precompute all valid bridge offset patterns.

        A bridge in Hex consists of:
        - Two stones that share exactly two common empty neighbors
        - Playing either empty neighbor maintains the virtual connection

        Returns list of: (partner_offset, carrier1_offset, carrier2_offset)
        where offsets are relative to a reference stone position.
        """
        # Define the 6 hex neighbor directions
        # (di, dj) offsets for each neighbor
        neighbors = [
            (-1, 0),   # top_left
            (-1, 1),   # top_right
            (0, -1),   # left
            (0, 1),    # right
            (1, -1),   # bot_left
            (1, 0),    # bot_right
        ]

        def neighbor_cells(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            i, j = pos
            return [(i + di, j + dj) for di, dj in neighbors]

        origin = (0, 0)
        origin_neighbors = set(neighbor_cells(origin))

        bridge_patterns = []

        # For each unordered pair of neighbor directions
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

                # Ensure the two carriers ARE exactly the common neighbors
                if {carrier1, carrier2} != common:
                    continue

                bridge_patterns.append((partner, carrier1, carrier2))

        return bridge_patterns

    # ========== BRIDGE DETECTION ==========

    def find_bridges(self, state: GameState, piece_type: str) -> List[Dict]:
        """
        Find all bridge structures for a given player.

        A bridge is a pair of stones with two shared empty carrier cells.

        Returns:
            List of bridge dictionaries with:
            - 'stone1': (i, j) position of first stone
            - 'stone2': (i, j) position of second stone
            - 'carrier1': (i, j) first carrier (empty cell)
            - 'carrier2': (i, j) second carrier (empty cell)
            - 'complete': True if both carriers are empty (unbreakable)
            - 'threatened': True if one carrier is occupied by opponent
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        opp_type = "B" if piece_type == "R" else "R"

        bridges = []
        seen_pairs = set()  # Avoid duplicate bridges

        # For each stone of the given type
        for (i, j), piece in env.items():
            if piece.get_type() != piece_type:
                continue

            # Check each bridge pattern
            for partner_off, carrier1_off, carrier2_off in self._bridge_offsets:
                # Calculate partner and carrier positions
                partner = (i + partner_off[0], j + partner_off[1])
                carrier1 = (i + carrier1_off[0], j + carrier1_off[1])
                carrier2 = (i + carrier2_off[0], j + carrier2_off[1])

                # Skip if positions are out of bounds
                if not self._in_bounds(partner, dim) or \
                   not self._in_bounds(carrier1, dim) or \
                   not self._in_bounds(carrier2, dim):
                    continue

                # Check if partner has our stone
                if partner not in env or env[partner].get_type() != piece_type:
                    continue

                # Create a canonical pair representation (smaller first)
                pair_key = (min((i, j), partner), max((i, j), partner))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Analyze carrier cells
                c1_status = self._cell_status(carrier1, env, piece_type)
                c2_status = self._cell_status(carrier2, env, piece_type)

                # Determine bridge status
                complete = (c1_status == "empty" and c2_status == "empty")
                threatened = (c1_status == "opponent" or c2_status == "opponent")
                secured = (c1_status == "friendly" or c2_status == "friendly")

                # Only count as bridge if at least one carrier is usable
                if c1_status != "opponent" or c2_status != "opponent":
                    bridges.append({
                        'stone1': (i, j),
                        'stone2': partner,
                        'carrier1': carrier1,
                        'carrier2': carrier2,
                        'c1_status': c1_status,
                        'c2_status': c2_status,
                        'complete': complete,
                        'threatened': threatened,
                        'secured': secured,
                    })

        return bridges

    def _in_bounds(self, pos: Tuple[int, int], dim: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= pos[0] < dim and 0 <= pos[1] < dim

    def _cell_status(self, pos: Tuple[int, int], env: Dict, piece_type: str) -> str:
        """
        Get the status of a cell.
        Returns: 'empty', 'friendly', or 'opponent'
        """
        if pos not in env:
            return "empty"
        elif env[pos].get_type() == piece_type:
            return "friendly"
        else:
            return "opponent"

    # ========== THREATENED BRIDGE DETECTION ==========

    def find_threatened_bridges(self, state: GameState) -> List[Dict]:
        """
        Find OUR bridges that are under attack (opponent in one carrier).

        These MUST be defended immediately by playing the other carrier.

        Returns:
            List of threatened bridge dicts with 'response' key indicating
            the cell we MUST play to maintain the connection.
        """
        bridges = self.find_bridges(state, self.piece_type)
        threatened = []

        for bridge in bridges:
            if bridge['threatened'] and not bridge['secured']:
                # One carrier has opponent, one is still empty
                # Find the empty carrier - that's our required response
                if bridge['c1_status'] == 'empty':
                    response = bridge['carrier1']
                elif bridge['c2_status'] == 'empty':
                    response = bridge['carrier2']
                else:
                    continue  # Both blocked, bridge is lost

                threatened.append({
                    **bridge,
                    'response': response,
                    'urgency': self._calculate_bridge_urgency(bridge, state)
                })

        # Sort by urgency (highest first)
        threatened.sort(key=lambda x: -x['urgency'])
        return threatened

    def _calculate_bridge_urgency(self, bridge: Dict, state: GameState) -> float:
        """
        Calculate how urgent it is to defend this bridge.

        Higher urgency for:
        - Bridges on our shortest path
        - Bridges closer to goal
        - Bridges that are part of a chain
        """
        dim = state.get_rep().get_dimensions()[0]
        s1, s2 = bridge['stone1'], bridge['stone2']

        urgency = 10.0  # Base urgency - defending bridges is important

        # Bonus for bridges advancing toward goal
        if self.piece_type == "R":
            avg_row = (s1[0] + s2[0]) / 2.0
            progress = avg_row / (dim - 1)
            urgency += progress * 5.0  # More urgent if closer to winning
        else:
            avg_col = (s1[1] + s2[1]) / 2.0
            progress = avg_col / (dim - 1)
            urgency += progress * 5.0

        return urgency

    def find_urgent_bridge_responses(self, state: GameState) -> List[Tuple[Tuple[int, int], float]]:
        """
        Find positions we MUST play to defend our threatened bridges.

        Returns list of (position, urgency) tuples.
        These should be TOP PRIORITY moves.
        """
        threatened = self.find_threatened_bridges(state)
        responses = []

        for bridge in threatened:
            responses.append((bridge['response'], bridge['urgency']))

        return responses

    def find_own_complete_bridge_carriers(self, state: GameState) -> Set[Tuple[int, int]]:
        """
        Find carrier positions of our COMPLETE bridges.

        These are positions we should AVOID playing in - the bridge is already
        a virtual connection. Playing here wastes a move.

        Only fill these if opponent intrudes (handled by find_urgent_bridge_responses).
        """
        bridges = self.find_bridges(state, self.piece_type)
        carriers_to_avoid = set()

        for bridge in bridges:
            if bridge['complete']:
                # Both carriers empty - don't play here!
                carriers_to_avoid.add(bridge['carrier1'])
                carriers_to_avoid.add(bridge['carrier2'])

        return carriers_to_avoid

    # ========== BRIDGE SCORING ==========

    def score_bridges(self, state: GameState, piece_type: str) -> float:
        """
        Score the bridge structures for a player.

        REACTIVE BRIDGE STRATEGY:
        - Complete bridges (both carriers empty): BEST - virtual connection, no move needed
        - Secured bridges (one carrier friendly): Good - physically connected
        - Threatened bridges (opponent in one carrier): BAD - must respond or lose!
        
        DIRECTIONAL STRATEGY:
        - Red (TOP→BOTTOM): Bridges with larger row span are more valuable
        - Blue (LEFT→RIGHT): Bridges with larger column span are more valuable
        - Goal-aligned bridges get 1.5x multiplier, perpendicular bridges get 0.7x
        """
        bridges = self.find_bridges(state, piece_type)
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0

        total_score = 0.0

        for bridge in bridges:
            score = 0.0

            # Base score for bridge status
            if bridge['complete']:
                # BEST: Virtual connection - no move needed to maintain!
                # This is the ideal state for a bridge
                score += 4.0
            elif bridge['secured']:
                # GOOD: One carrier is ours - physically connected
                score += 3.0
            elif bridge['threatened']:
                # BAD: Opponent has intruded - we MUST respond
                # Don't add points here - the threat_penalty in heuristic handles this
                score += 0.0  # Threatened bridges aren't worth points until defended

            # Calculate directional alignment
            s1, s2 = bridge['stone1'], bridge['stone2']
            row_span = abs(s1[0] - s2[0])
            col_span = abs(s1[1] - s2[1])
            
            # Directional multiplier based on goal alignment
            if piece_type == "R":
                # Red wants vertical (top-bottom) progress
                # Bridges with larger row span are better
                if row_span > col_span:
                    directional_multiplier = 1.5  # Goal-aligned
                elif row_span < col_span:
                    directional_multiplier = 0.7  # Perpendicular to goal
                else:
                    directional_multiplier = 1.0  # Diagonal
            else:  # Blue
                # Blue wants horizontal (left-right) progress
                # Bridges with larger column span are better
                if col_span > row_span:
                    directional_multiplier = 1.5  # Goal-aligned
                elif col_span < row_span:
                    directional_multiplier = 0.7  # Perpendicular to goal
                else:
                    directional_multiplier = 1.0  # Diagonal
            
            # Apply directional multiplier to base score
            score *= directional_multiplier

            # Bonus for bridges advancing toward goal
            if piece_type == "R":
                avg_row = (s1[0] + s2[0]) / 2.0
                progress = avg_row / (dim - 1)
                score += progress * 0.5 + row_span * 0.3
            else:
                avg_col = (s1[1] + s2[1]) / 2.0
                progress = avg_col / (dim - 1)
                score += progress * 0.5 + col_span * 0.3

            # Bonus for bridges near center (decays as game progresses)
            # In opening, center bridges are very valuable
            # In endgame, position matters less than connections
            center_dist = (abs(s1[0] - center) + abs(s1[1] - center) +
                           abs(s2[0] - center) + abs(s2[1] - center)) / 2.0

            # Use dynamic center weight for bridge center bonus
            dynamic_center = self.get_dynamic_center_weight(state)
            center_bonus = (dynamic_center / self.W_CENTER_BASE) * 2.0 / (1.0 + center_dist * 0.1)
            score += center_bonus * 0.5

            total_score += score

        return total_score

    # ========== LOCAL CONNECTIVITY SCORING ==========

    def local_connectivity_score(self, state: GameState, pos: Tuple[int, int], 
                                  piece_type: str) -> float:
        """
        Score how well a move at pos connects to our existing stones.
        
        Higher scores for:
        - Moves adjacent to our pieces
        - Moves that can form bridges with our pieces
        - Moves that unify disconnected groups
        
        Args:
            state: Current game state
            pos: Position to evaluate
            piece_type: Our piece type
            
        Returns:
            Connectivity score (0.0 to 10.0+)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        i, j = pos
        
        if pos in env:
            return 0.0  # Position already occupied
        
        score = 0.0
        
        # Count direct friendly neighbors (immediate connection)
        neighbors = state.get_neighbours(i, j)
        friendly_neighbors = []
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == piece_type:
                friendly_neighbors.append((ni, nj))
                score += 2.0  # Strong bonus for direct connection
        
        # Bonus for connecting multiple groups (unification)
        if len(friendly_neighbors) >= 2:
            score += 3.0  # Unifying groups is valuable
        
        # Check potential bridges we could form
        bridge_potential = 0.0
        for partner_off, c1_off, c2_off in self._bridge_offsets:
            partner = (i + partner_off[0], j + partner_off[1])
            carrier1 = (i + c1_off[0], j + c1_off[1])
            carrier2 = (i + c2_off[0], j + c2_off[1])
            
            if not self._in_bounds(partner, dim):
                continue
            
            # Check if partner has our stone
            if partner in env and env[partner].get_type() == piece_type:
                # Check if carriers are available
                if self._in_bounds(carrier1, dim) and self._in_bounds(carrier2, dim):
                    c1_empty = carrier1 not in env
                    c2_empty = carrier2 not in env
                    
                    if c1_empty and c2_empty:
                        bridge_potential += 1.5  # Can form complete bridge
                    elif c1_empty or c2_empty:
                        bridge_potential += 0.8  # Can form partial bridge
        
        score += bridge_potential
        
        # Bonus for connecting toward goal
        if piece_type == "R":
            # Red wants to connect vertically (top to bottom)
            # Check if we're between our pieces vertically
            has_above = any(ni < i for ni, nj in friendly_neighbors)
            has_below = any(ni > i for ni, nj in friendly_neighbors)
            if has_above and has_below:
                score += 2.0  # Connecting vertical chain
        else:  # Blue
            # Blue wants to connect horizontally (left to right)
            has_left = any(nj < j for ni, nj in friendly_neighbors)
            has_right = any(nj > j for ni, nj in friendly_neighbors)
            if has_left and has_right:
                score += 2.0  # Connecting horizontal chain
        
        return score

    # ========== BRIDGE CUT POINT DETECTION ==========

    def find_cut_points(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]:
        """
        ENHANCED: Find optimal cut points to block opponent bridges.
        
        Now includes:
        - Bridge cutting (original)
        - Vital point detection (blocks multiple paths)
        - Corridor factor (prioritize cuts in opponent's main corridor)
        - Connectivity bonus (prefer blocks that also help our position)

        Returns list of (position, priority) tuples where:
        - position: the cell to play to cut the bridge
        - priority: how important this cut is (higher = more urgent)
        """
        bridges = self.find_bridges(state, opponent_type)
        dim = state.get_rep().get_dimensions()[0]
        piece_type = "B" if opponent_type == "R" else "R"

        cut_points = {}  # pos -> best priority

        for bridge in bridges:
            if not bridge['complete']:
                # Bridge already has one carrier blocked - less urgent
                continue

            c1, c2 = bridge['carrier1'], bridge['carrier2']
            s1, s2 = bridge['stone1'], bridge['stone2']

            # Calculate base priority based on how dangerous this bridge is
            base_priority = 1.0

            # Higher priority for bridges advancing toward goal
            if opponent_type == "R":
                # Red bridges advancing downward are more dangerous
                avg_row = (s1[0] + s2[0]) / 2.0
                progress = avg_row / (dim - 1)
                base_priority += progress * 3.0
            else:
                # Blue bridges advancing rightward are more dangerous
                avg_col = (s1[1] + s2[1]) / 2.0
                progress = avg_col / (dim - 1)
                base_priority += progress * 3.0

            # Process both carrier positions
            for carrier_pos in [c1, c2]:
                priority = base_priority
                
                # ENHANCEMENT 1: Vital point detection
                vital_score = self.detect_vital_point(state, carrier_pos, piece_type, opponent_type)
                priority += vital_score * self.W_VITAL_POINT / 10.0  # Scale to reasonable range
                
                # ENHANCEMENT 2: Corridor factor
                in_corridor, corridor_strength = self.is_in_opponent_corridor(state, carrier_pos, opponent_type)
                if in_corridor:
                    corridor_multiplier = 1.0 + (corridor_strength * (self.W_CORRIDOR - 1.0))
                    priority *= corridor_multiplier
                else:
                    # Not in corridor - less valuable cut
                    priority *= 0.7
                
                # ENHANCEMENT 3: Connectivity bonus (dual-purpose moves)
                connectivity = self.local_connectivity_score(state, carrier_pos, piece_type)
                priority += connectivity * self.W_CONNECTIVITY / 10.0  # Scale appropriately
                
                # ENHANCEMENT 4: Ladder break detection
                ladder_break_value = self.detect_ladder_break_for_block(
                    state, carrier_pos, piece_type, opponent_type
                )
                priority += ladder_break_value * self.W_LADDER_BREAK / 10.0
                
                # Update best priority for this position
                if carrier_pos not in cut_points or priority > cut_points[carrier_pos]:
                    cut_points[carrier_pos] = priority

        # Convert to list and sort by priority
        result = [(pos, priority) for pos, priority in cut_points.items()]
        result.sort(key=lambda x: -x[1])
        
        return result

    def detect_vital_point(self, state: GameState, pos: Tuple[int, int], 
                          piece_type: str, opponent_type: str) -> float:
        """
        Detect if a position is a vital point for blocking opponent.
        
        A vital point is a position that:
        - Appears on multiple opponent shortest paths (chokepoint)
        - Blocks key intersection in opponent's path network
        - Forces opponent to take longer alternative routes
        
        Args:
            state: Current game state
            pos: Position to evaluate
            piece_type: Our piece type
            opponent_type: Opponent's piece type
            
        Returns:
            Vital point score (0.0 to 10.0+, higher = more critical)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        if pos in env:
            return 0.0  # Already occupied
        
        # Use Dijkstra to track how many shortest paths go through this position
        # We'll run path finding twice: once with pos blocked, once without
        
        # First: Calculate opponent's current shortest path
        current_dist = self.shortest_path_distance(state, opponent_type, consider_bridges=False)
        
        # Second: Simulate blocking this position and recalculate
        # Create temporary state with our piece at pos
        simulated_env = env.copy()
        from seahorse.game.representation import Piece
        simulated_env[pos] = Piece(piece_type)
        
        # Calculate path distance if we block here
        # We need to create a modified state - let's use a simpler approach
        # Check how many opponent neighbors this position has
        neighbors = state.get_neighbours(pos[0], pos[1])
        opp_neighbor_count = 0
        opp_neighbors = []
        
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == opponent_type:
                opp_neighbor_count += 1
                opp_neighbors.append((ni, nj))
        
        vital_score = 0.0
        
        # High score if blocking between multiple opponent pieces
        if opp_neighbor_count >= 2:
            vital_score += opp_neighbor_count * 2.0  # Critical intersection
        
        # Check if position is on opponent's main axis
        if opponent_type == "R":
            # Red moves top to bottom
            # Vital points are in middle rows
            row_progress = pos[0] / (dim - 1)
            if 0.3 < row_progress < 0.7:
                vital_score += 2.0  # In critical middle zone
        else:  # Blue
            # Blue moves left to right
            col_progress = pos[1] / (dim - 1)
            if 0.3 < col_progress < 0.7:
                vital_score += 2.0  # In critical middle zone
        
        # Bonus if position is between two opponent groups
        # Check if opponent neighbors are not connected to each other
        if len(opp_neighbors) >= 2:
            # Check if the opponent pieces are separated
            opp_neighbor_set = set(opp_neighbors)
            separated = True
            for opp_pos in opp_neighbors:
                opp_neighs = state.get_neighbours(opp_pos[0], opp_pos[1])
                for _, (n_type, n_pos) in opp_neighs.values():
                    if n_type == opponent_type and n_pos in opp_neighbor_set and n_pos != opp_pos:
                        separated = False
                        break
                if not separated:
                    break
            
            if separated:
                vital_score += 4.0  # Blocking bridge between separated groups!
        
        # Check if blocking here creates a ladder break
        ladder_break_value = self._check_ladder_break(state, pos, piece_type, opponent_type)
        vital_score += ladder_break_value
        
        return vital_score

    def _check_ladder_break(self, state: GameState, pos: Tuple[int, int],
                           piece_type: str, opponent_type: str) -> float:
        """
        Check if playing at pos breaks an opponent ladder.
        
        Returns:
            Ladder break value (0.0 if no ladder, positive if breaks ladder)
        """
        # Simplified ladder break detection
        # A ladder break typically stops a forcing sequence
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        # Check if opponent has a forcing sequence toward goal that we interrupt
        neighbors = state.get_neighbours(pos[0], pos[1])
        
        # Look for opponent stones that are part of a potential ladder
        opp_count = 0
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == opponent_type:
                opp_count += 1
        
        # If we're adjacent to 2+ opponent pieces near their goal, likely breaking ladder
        if opp_count >= 2:
            if opponent_type == "R":
                # Check if we're in front of their advance (blocking downward progress)
                progress = pos[0] / (dim - 1)
                if progress > 0.4:  # They're advancing, we're blocking
                    return 3.0
            else:  # Blue
                progress = pos[1] / (dim - 1)
                if progress > 0.4:
                    return 3.0
        
        return 0.0

    def find_path_cut_points(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]:
        """
        Find strategic cut points on opponent's shortest path(s).
        
        Unlike bridge blocking, this identifies cells that are critical to
        opponent's winning path, even if they don't have bridges there yet.
        
        Strategy:
        - Run modified Dijkstra to track ALL cells that lie on shortest paths
        - Identify "chokepoints" that appear on multiple paths
        - Prioritize cells that are:
          1. On the shortest path
          2. Close to opponent's current progress
          3. On multiple alternative paths (chokepoints)
        
        Returns list of (position, priority) tuples sorted by blocking priority.
        """
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        
        # Track distances and parent cells for path reconstruction
        distances = {}
        parents = {}  # Cell -> list of parent cells with same distance
        pq = []
        
        # Initialize starting positions for opponent
        if opponent_type == "R":
            # Red starts from top row
            for j in range(dim):
                if (0, j) in env and env[(0, j)].get_type() == opponent_type:
                    cost = 0
                elif (0, j) not in env:
                    cost = 1
                else:
                    continue  # Our piece, skip
                distances[(0, j)] = cost
                parents[(0, j)] = []
                heapq.heappush(pq, (cost, 0, j))
        else:  # Blue
            # Blue starts from left column
            for i in range(dim):
                if (i, 0) in env and env[(i, 0)].get_type() == opponent_type:
                    cost = 0
                elif (i, 0) not in env:
                    cost = 1
                else:
                    continue
                distances[(i, 0)] = cost
                parents[(i, 0)] = []
                heapq.heappush(pq, (cost, i, 0))
        
        # Dijkstra with path tracking
        goal_cells = set()
        min_goal_dist = float('inf')
        
        while pq:
            dist, i, j = heapq.heappop(pq)
            
            if (i, j) in distances and dist > distances[(i, j)]:
                continue
            
            # Check if reached goal
            is_goal = (opponent_type == "R" and i == dim - 1) or \
                     (opponent_type == "B" and j == dim - 1)
            
            if is_goal:
                if dist < min_goal_dist:
                    min_goal_dist = dist
                    goal_cells = {(i, j)}
                elif dist == min_goal_dist:
                    goal_cells.add((i, j))
                continue
            
            # Explore neighbors
            neighbors = state.get_neighbours(i, j)
            for neighbor_type, (ni, nj) in neighbors.values():
                if neighbor_type == "OUTSIDE":
                    continue
                
                # Calculate cost
                if neighbor_type == opponent_type:
                    new_dist = dist
                elif neighbor_type == "EMPTY":
                    new_dist = dist + 1
                else:
                    continue  # Our piece - can't use
                
                # Update distance and parents
                if (ni, nj) not in distances or new_dist < distances[(ni, nj)]:
                    distances[(ni, nj)] = new_dist
                    parents[(ni, nj)] = [(i, j)]
                    heapq.heappush(pq, (new_dist, ni, nj))
                elif new_dist == distances[(ni, nj)]:
                    # Multiple shortest paths through this cell
                    if (i, j) not in parents[(ni, nj)]:
                        parents[(ni, nj)].append((i, j))
        
        # Backtrack from goal cells to find all cells on shortest paths
        path_cells = set()
        path_frequency = {}  # How many times each cell appears in paths
        
        def backtrack(cell):
            """Recursively backtrack to mark all cells on shortest paths."""
            if cell in path_cells:
                return
            path_cells.add(cell)
            path_frequency[cell] = path_frequency.get(cell, 0) + 1
            
            if cell in parents:
                for parent in parents[cell]:
                    backtrack(parent)
        
        for goal_cell in goal_cells:
            backtrack(goal_cell)
        
        # ENHANCED: Score each empty cell on the path with additional factors
        piece_type = "B" if opponent_type == "R" else "R"
        cut_points = []
        
        for cell in path_cells:
            if cell not in env:  # Only consider empty cells
                i, j = cell
                priority = 0.0
                
                # ORIGINAL: Higher priority for cells appearing in more paths (chokepoints)
                frequency = path_frequency.get(cell, 1)
                priority += frequency * 2.0
                
                # ORIGINAL: Higher priority for cells closer to opponent's progress
                if opponent_type == "R":
                    # Find how far down the board opponent has progressed
                    opp_progress = max((pos[0] for pos, p in env.items() 
                                       if p.get_type() == opponent_type), default=0)
                    # Prioritize cells near the opponent's current front
                    dist_from_front = abs(i - opp_progress)
                    priority += max(0, 5.0 - dist_from_front * 0.5)
                else:  # Blue
                    opp_progress = max((pos[1] for pos, p in env.items() 
                                       if p.get_type() == opponent_type), default=0)
                    dist_from_front = abs(j - opp_progress)
                    priority += max(0, 5.0 - dist_from_front * 0.5)
                
                # ORIGINAL: Higher priority for cells closer to goal (more urgent)
                if cell in distances:
                    remaining_dist = distances[cell]
                    # Closer to completion = more urgent
                    urgency = (dim - remaining_dist) / dim
                    priority += urgency * 3.0
                
                # ENHANCEMENT 1: Corridor detection (already on path, boost if high strength)
                # Since we already know cell is on a shortest path, check corridor strength
                _, corridor_strength = self.is_in_opponent_corridor(state, cell, opponent_type)
                if corridor_strength > 0.5:
                    # High corridor strength = critical chokepoint
                    priority *= (1.0 + corridor_strength * 0.5)  # Boost up to 1.75x
                
                # ENHANCEMENT 2: Ladder break detection
                ladder_break_value = self.detect_ladder_break_for_block(
                    state, cell, piece_type, opponent_type
                )
                priority += ladder_break_value * self.W_LADDER_BREAK / 10.0
                
                # ENHANCEMENT 3: Connectivity scoring (dual-purpose moves)
                connectivity = self.local_connectivity_score(state, cell, piece_type)
                priority += connectivity * self.W_CONNECTIVITY / 10.0
                
                # ENHANCEMENT 4: Vital point bonus (if it blocks multiple paths)
                if frequency >= 2:  # Already identified as chokepoint
                    vital_score = self.detect_vital_point(state, cell, piece_type, opponent_type)
                    priority += vital_score * self.W_VITAL_POINT / 10.0
                
                cut_points.append((cell, priority))
        
        # Sort by priority (highest first)
        cut_points.sort(key=lambda x: -x[1])
        
        return cut_points

    def is_in_opponent_corridor(self, state: GameState, pos: Tuple[int, int], 
                                opponent_type: str) -> Tuple[bool, float]:
        """
        Check if a position is in opponent's main corridor (critical path zone).
        
        The corridor is the set of cells that lie on or near the opponent's
        shortest paths to their goal. Blocking in the corridor is most effective.
        
        Args:
            state: Current game state
            pos: Position to check
            opponent_type: Opponent's piece type
            
        Returns:
            Tuple of (is_in_corridor: bool, corridor_strength: float)
            - is_in_corridor: True if position is on a shortest path
            - corridor_strength: 0.0 to 1.0, how central to corridor (1.0 = critical)
        """
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        
        if pos in env:
            return False, 0.0  # Already occupied
        
        # Run Dijkstra from opponent's start to track all shortest paths
        distances = {}
        parents = {}  # For path reconstruction
        pq = []
        
        # Initialize from opponent's starting edge
        if opponent_type == "R":
            for j in range(dim):
                if (0, j) in env and env[(0, j)].get_type() == opponent_type:
                    cost = 0
                elif (0, j) not in env:
                    cost = 1
                else:
                    continue
                distances[(0, j)] = cost
                parents[(0, j)] = []
                heapq.heappush(pq, (cost, 0, j))
        else:  # Blue
            for i in range(dim):
                if (i, 0) in env and env[(i, 0)].get_type() == opponent_type:
                    cost = 0
                elif (i, 0) not in env:
                    cost = 1
                else:
                    continue
                distances[(i, 0)] = cost
                parents[(i, 0)] = []
                heapq.heappush(pq, (cost, i, 0))
        
        # Dijkstra to find shortest path distances
        min_goal_dist = float('inf')
        goal_cells = set()
        
        while pq:
            dist, i, j = heapq.heappop(pq)
            
            if (i, j) in distances and dist > distances[(i, j)]:
                continue
            
            # Check if reached goal
            is_goal = (opponent_type == "R" and i == dim - 1) or \
                     (opponent_type == "B" and j == dim - 1)
            
            if is_goal:
                if dist < min_goal_dist:
                    min_goal_dist = dist
                    goal_cells = {(i, j)}
                elif dist == min_goal_dist:
                    goal_cells.add((i, j))
                continue
            
            # Explore neighbors
            neighbors = state.get_neighbours(i, j)
            for neighbor_type, (ni, nj) in neighbors.values():
                if neighbor_type == "OUTSIDE":
                    continue
                
                if neighbor_type == opponent_type:
                    new_dist = dist
                elif neighbor_type == "EMPTY":
                    new_dist = dist + 1
                else:
                    continue  # Our piece
                
                if (ni, nj) not in distances or new_dist < distances[(ni, nj)]:
                    distances[(ni, nj)] = new_dist
                    parents[(ni, nj)] = [(i, j)]
                    heapq.heappush(pq, (new_dist, ni, nj))
                elif new_dist == distances[(ni, nj)]:
                    if (i, j) not in parents[(ni, nj)]:
                        parents[(ni, nj)].append((i, j))
        
        # Check if pos is on any shortest path
        if pos not in distances:
            return False, 0.0  # Not reachable
        
        # Backtrack from goals to find all cells on shortest paths
        path_cells = set()
        
        def backtrack(cell):
            if cell in path_cells:
                return
            path_cells.add(cell)
            if cell in parents:
                for parent in parents[cell]:
                    backtrack(parent)
        
        for goal in goal_cells:
            backtrack(goal)
        
        if pos not in path_cells:
            # Not on shortest path, but might be near it
            # Check distance to nearest path cell
            min_dist_to_path = float('inf')
            for path_cell in path_cells:
                dist = abs(pos[0] - path_cell[0]) + abs(pos[1] - path_cell[1])
                min_dist_to_path = min(min_dist_to_path, dist)
            
            if min_dist_to_path <= 1:
                # Adjacent to corridor
                return True, 0.3
            else:
                return False, 0.0
        
        # Position IS on shortest path - calculate corridor strength
        # Strength based on:
        # 1. How many paths go through this cell (chokepoint factor)
        # 2. Distance from goal (closer = more critical)
        
        pos_dist = distances[pos]
        remaining_dist = min_goal_dist - pos_dist
        
        # Cells closer to goal are more critical
        goal_proximity = 1.0 - (remaining_dist / max(1, min_goal_dist))
        
        # Count how many paths go through this cell (via parent counting)
        path_count = len(parents.get(pos, [])) + 1
        chokepoint_factor = min(1.0, path_count / 3.0)
        
        # Combine factors
        corridor_strength = (goal_proximity * 0.6 + chokepoint_factor * 0.4)
        
        return True, corridor_strength

    # ========== BRIDGE OPPORTUNITY DETECTION ==========

    def find_bridge_opportunities(self, state: GameState, piece_type: str) -> List[Tuple[Tuple[int, int], float]]:
        """
        Find moves that would create new bridges or extend existing ones.

        Returns list of (position, value) tuples where:
        - position: the cell to play to form a bridge
        - value: how valuable this bridge formation is
        
        Prioritizes goal-aligned bridges:
        - Red: bridges with larger row span (vertical)
        - Blue: bridges with larger column span (horizontal)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0

        opportunities = []

        # For each empty cell
        for i in range(dim):
            for j in range(dim):
                if (i, j) in env:
                    continue  # Cell occupied

                # Check if playing here would form bridges
                bridge_value = 0.0

                for partner_off, carrier1_off, carrier2_off in self._bridge_offsets:
                    # Calculate what would be the partner stone position
                    # if this empty cell becomes a bridge carrier
                    # Actually, we need to check if playing at (i,j) creates a stone
                    # that forms a bridge with an existing stone

                    # Check pattern where (i,j) is the new stone
                    for partner_off2, c1_off2, c2_off2 in self._bridge_offsets:
                        partner = (i + partner_off2[0], j + partner_off2[1])
                        carrier1 = (i + c1_off2[0], j + c1_off2[1])
                        carrier2 = (i + c2_off2[0], j + c2_off2[1])

                        if not self._in_bounds(partner, dim):
                            continue

                        # Check if partner has our stone
                        if partner in env and env[partner].get_type() == piece_type:
                            # Would form a bridge! Check carriers
                            if self._in_bounds(carrier1, dim) and self._in_bounds(carrier2, dim):
                                c1_empty = carrier1 not in env
                                c2_empty = carrier2 not in env

                                if c1_empty and c2_empty:
                                    # Complete bridge formation
                                    value = 2.0
                                elif c1_empty or c2_empty:
                                    # Partial bridge (one carrier available)
                                    value = 1.0
                                else:
                                    continue

                                # Calculate directional alignment
                                row_span = abs(i - partner[0])
                                col_span = abs(j - partner[1])
                                
                                # Apply directional multiplier
                                if piece_type == "R":
                                    # Red wants vertical bridges
                                    if row_span > col_span:
                                        value *= 1.5  # Goal-aligned
                                    elif row_span < col_span:
                                        value *= 0.7  # Perpendicular
                                else:  # Blue
                                    # Blue wants horizontal bridges
                                    if col_span > row_span:
                                        value *= 1.5  # Goal-aligned
                                    elif col_span < row_span:
                                        value *= 0.7  # Perpendicular

                                # Bonus for goal-direction bridges
                                if piece_type == "R":
                                    progress = (i + partner[0]) / (2.0 * (dim - 1))
                                else:
                                    progress = (j + partner[1]) / (2.0 * (dim - 1))
                                value += progress * 0.5

                                # Bonus for center proximity
                                center_dist = abs(i - center) + abs(j - center)
                                value += 0.5 / (1.0 + center_dist * 0.1)

                                bridge_value += value

                if bridge_value > 0:
                    opportunities.append(((i, j), bridge_value))

        # Sort by value (highest first)
        opportunities.sort(key=lambda x: -x[1])
        return opportunities

    # ========== FAST HEURISTIC (for rollouts) ==========

    def fast_heuristic(self, state: GameState) -> float:
        """
        Fast position evaluation for rollouts.
        Called thousands of times, must be VERY fast.

        Components:
        - Center control: DECAYS exponentially as game progresses
        - Progress: pieces advancing toward goal are better
        - Quick bridge count: approximate bridge strength
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

        # Get dynamic center weight (decays as game progresses)
        center_weight = self.get_dynamic_center_weight(state)

        my_score = 0.0
        opp_score = 0.0

        # Track stone positions for quick bridge detection
        my_stones = []
        opp_stones = []

        for (i, j), piece in env.items():
            # Center bonus: pieces closer to center are more valuable
            # Weight decays exponentially during game
            dist_from_center = abs(i - center) + abs(j - center)
            center_value = center_weight / (1.0 + dist_from_center * 0.1)

            # Progress bonus: advancing toward goal
            if piece.get_type() == self.piece_type:
                if self.piece_type == "R":
                    progress = i / (dim - 1)  # Advance from top to bottom
                else:  # Blue
                    progress = j / (dim - 1)  # Advance from left to right

                my_score += center_value + progress * self.W_PROGRESS
                my_stones.append((i, j))

            else:  # Opponent's piece
                if self.opp_type == "R":
                    progress = i / (dim - 1)
                else:
                    progress = j / (dim - 1)

                opp_score += center_value + progress * self.W_PROGRESS
                opp_stones.append((i, j))

        # Quick bridge approximation (check diagonal adjacency patterns)
        my_bridge_bonus = self._quick_bridge_count(my_stones, env, self.piece_type, dim)
        opp_bridge_bonus = self._quick_bridge_count(opp_stones, env, self.opp_type, dim)

        my_score += my_bridge_bonus * 1.5
        opp_score += opp_bridge_bonus * 1.5

        return my_score - opp_score

    def _quick_bridge_count(self, stones: List[Tuple[int, int]], env: Dict,
                            piece_type: str, dim: int) -> float:
        """
        Quick bridge counting for fast heuristic.
        Only checks the most common bridge pattern (diagonal).
        """
        count = 0.0
        stone_set = set(stones)

        for (i, j) in stones:
            # Check diagonal pattern: (i,j) to (i+1,j+1)
            partner = (i + 1, j + 1)
            if partner in stone_set:
                carrier1 = (i, j + 1)
                carrier2 = (i + 1, j)
                if self._in_bounds(carrier1, dim) and self._in_bounds(carrier2, dim):
                    c1_free = carrier1 not in env
                    c2_free = carrier2 not in env
                    if c1_free and c2_free:
                        count += 1.0
                    elif c1_free or c2_free:
                        count += 0.5

            # Check other diagonal: (i,j) to (i-1,j-1)
            partner = (i - 1, j - 1)
            if partner in stone_set:
                carrier1 = (i - 1, j)
                carrier2 = (i, j - 1)
                if self._in_bounds(carrier1, dim) and self._in_bounds(carrier2, dim):
                    c1_free = carrier1 not in env
                    c2_free = carrier2 not in env
                    if c1_free and c2_free:
                        count += 1.0
                    elif c1_free or c2_free:
                        count += 0.5

        return count / 2.0  # Divide by 2 to avoid double counting

    # ========== ACCURATE HEURISTIC (for root move ordering) ==========

    def accurate_heuristic(self, state: GameState) -> float:
        """
        ENHANCED: More accurate but slower position evaluation.
        Used for ordering root moves (called ~100 times per turn).

        Components (weighted):
        1. Shortest path distance (most important)
        2. Bridge structures (complete bridges = virtual connections)
        3. PENALTY for undefended threatened bridges (critical!)
        4. ENHANCED Blocking opponent - includes:
           - Bridge cuts (blocking opponent bridges)
           - Path cuts (blocking opponent's shortest paths)
           - Vital points (blocking multiple paths simultaneously)
           - Corridor control (disrupting opponent's main corridor)
           - Ladder breaks (stopping forcing sequences)
           - Connectivity bonus (dual-purpose moves)
        5. Connectivity evaluation (how well our stones work together)
        6. Center control (decays exponentially as game progresses)

        Key: Complete bridges count as connections without needing to fill them.
        Threatened bridges that aren't defended are a major problem.
        Enhanced blocking now considers strategic factors beyond simple bridge cutting.
        """
        # Terminal state
        if state.is_done():
            if state.get_player_score(self) > 0:
                return 1000.0
            else:
                return -1000.0

        # 1. Path distance (with virtual connections from complete bridges)
        my_dist = self.shortest_path_distance(state, self.piece_type, consider_bridges=True)
        opp_dist = self.shortest_path_distance(state, self.opp_type, consider_bridges=True)
        path_advantage = opp_dist - my_dist

        # 2. Bridge structures (complete bridges are valuable virtual connections)
        my_bridges = self.score_bridges(state, self.piece_type)
        opp_bridges = self.score_bridges(state, self.opp_type)
        bridge_advantage = my_bridges - opp_bridges

        # 3. CRITICAL: Penalize undefended threatened bridges
        # If we have threatened bridges, this position is dangerous!
        threatened = self.find_threatened_bridges(state)
        threat_penalty = len(threatened) * 15.0  # Significant penalty per threatened bridge

        # 4. ENHANCED Blocking score - includes vital points, corridor, connectivity
        # The enhanced methods now include vital points, ladder breaks, corridor factors
        bridge_cuts = self.find_cut_points(state, self.opp_type)
        path_cuts = self.find_path_cut_points(state, self.opp_type)
        
        # Sum up blocking urgency from both sources (already enhanced with vital points, etc.)
        bridge_block_urgency = sum(priority for _, priority in bridge_cuts[:3])
        path_block_urgency = sum(priority for _, priority in path_cuts[:3])
        block_urgency = bridge_block_urgency + path_block_urgency * 0.8  # Path cuts slightly less weighted
        
        # Dynamic blocking urgency: block MORE when opponent is closer to winning
        path_deficit = my_dist - opp_dist  # Positive = we're behind
        if path_deficit > 0:
            # We're behind - blocking becomes more urgent
            block_multiplier = 1.0 + (path_deficit * 0.15)  # Scale up blocking when behind
        else:
            block_multiplier = 1.0
        
        # 4b. ENHANCEMENT: Corridor control evaluation
        # Bonus if our pieces are in opponent's corridor (disrupting their plan)
        corridor_disruption = 0.0
        for (i, j), piece in env.items():
            if piece.get_type() == self.piece_type:
                in_corridor, strength = self.is_in_opponent_corridor(state, (i, j), self.opp_type)
                # Note: pos is occupied, but we check if it WAS in their corridor
                # Actually, we should check empty cells in their corridor for blocking potential
                # Skip this part as it's complex - the enhanced cut points already handle this
                pass
        
        # 4c. ENHANCEMENT: Connectivity evaluation
        # Reward well-connected positions (our stones work together)
        my_connectivity = 0.0
        stone_count = 0
        for (i, j), piece in env.items():
            if piece.get_type() == self.piece_type:
                # Count neighbors
                neighbors = state.get_neighbours(i, j)
                friendly_count = sum(1 for ntype, _ in neighbors.values() if ntype == self.piece_type)
                my_connectivity += friendly_count
                stone_count += 1
        
        # Normalize connectivity score
        if stone_count > 0:
            avg_connectivity = my_connectivity / stone_count
            connectivity_advantage = avg_connectivity * 2.0  # Scale to reasonable range
        else:
            connectivity_advantage = 0.0

        # 5. Center control (DECAYS exponentially as game progresses)
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0

        # Get dynamic center weight
        center_weight = self.get_dynamic_center_weight(state)

        my_center = 0.0
        opp_center = 0.0
        for (i, j), piece in env.items():
            dist = abs(i - center) + abs(j - center)
            value = 1.0 / (1.0 + dist * 0.1)
            if piece.get_type() == self.piece_type:
                my_center += value
            else:
                opp_center += value
        center_advantage = my_center - opp_center

        # ENHANCED Combined score with weights (center weight is dynamic!)
        # Note: block_urgency represents OPPONENT's threats - higher = we should block
        # The negative sign means: more opponent threats = worse position for us
        # Enhanced blocking now includes vital points, corridor factors, ladder breaks, connectivity
        score = (self.W_PATH * path_advantage +
                 self.W_BRIDGE * bridge_advantage -
                 threat_penalty -  # BIG penalty for undefended threats
                 self.W_BLOCK * block_urgency * 0.5 * block_multiplier +  # Enhanced blocking urgency
                 self.W_CONNECTIVITY * connectivity_advantage +  # NEW: Connectivity bonus
                 center_weight * center_advantage)  # Dynamic center weight!

        return score

    def shortest_path_distance(self, state: GameState, piece_type: str,
                                 consider_bridges: bool = True) -> float:
        """
        Calculate minimum number of empty cells needed to connect sides.
        Uses Dijkstra's algorithm with optional bridge awareness.

        Cost model:
        - Own pieces: 0 (already placed)
        - Empty cells: 1 (need to place a piece)
        - Bridge carriers: 0.5 (virtual connection, only need one)
        - Opponent pieces: infinity (cannot use)

        Args:
            state: Current game state
            piece_type: "R" or "B"
            consider_bridges: If True, bridges count as virtual connections
        """
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        visited = set()
        pq = []  # Priority queue: (distance, i, j)

        opp_type = "B" if piece_type == "R" else "R"

        # Find bridge-connected cells for virtual connection bonus
        bridge_connections = set()
        if consider_bridges:
            bridges = self.find_bridges(state, piece_type)
            for bridge in bridges:
                if bridge['complete'] or bridge['secured']:
                    # Mark both stones as virtually connected
                    bridge_connections.add((bridge['stone1'], bridge['stone2']))
                    bridge_connections.add((bridge['stone2'], bridge['stone1']))

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

            # Explore direct neighbors
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

            # Also explore bridge-connected cells (virtual connections)
            if consider_bridges:
                for (stone1, stone2) in bridge_connections:
                    if stone1 == (i, j) and stone2 not in visited:
                        # Can reach stone2 via bridge for free
                        heapq.heappush(pq, (dist, stone2[0], stone2[1]))

        # No path found (shouldn't happen in valid Hex)
        return dim

    def virtual_connection_distance(self, state: GameState, piece_type: str) -> float:
        """
        Calculate path distance considering virtual connections (bridges).

        Virtual connections are groups of cells that are effectively connected
        because the opponent cannot prevent the connection.

        This is more accurate than raw path distance for evaluating positions.
        """
        return self.shortest_path_distance(state, piece_type, consider_bridges=True)

    # ========== ROLLOUT POLICY ==========

    def biased_rollout(self, state: GameState) -> float:
        """
        Bridge-aware epsilon-greedy rollout.

        Uses a mix of:
        - Random exploration (epsilon)
        - Fast heuristic-guided exploitation
        - Occasional bridge-focused moves
        """
        s = state
        steps = 0
        max_depth = 80  # Prevent infinite loops

        while not s.is_done() and steps < max_depth:
            actions = list(s.get_possible_heavy_actions())
            if not actions:
                break

            # Dynamic epsilon: more exploration early, more exploitation later
            epsilon = 0.30 if steps < 5 else (0.20 if steps < 15 else 0.10)

            roll = random.random()

            if roll < epsilon:
                # Explore: random action
                action = random.choice(actions)

            elif roll < epsilon + 0.1 and steps < 20:
                # Bridge-biased move (10% of time early in rollout)
                # Quick check for bridge opportunities
                action = self._quick_bridge_move(s, actions)

            else:
                # Exploit: best action according to fast heuristic
                action = max(actions,
                             key=lambda a: self.fast_heuristic(a.get_next_game_state()))

            s = action.get_next_game_state()
            steps += 1

        # Return heuristic evaluation of final state
        return self.fast_heuristic(s)

    def _quick_bridge_move(self, state: GameState, actions: List) -> 'Action':
        """
        Quickly select a bridge-forming move if available.

        IMPORTANT: Avoids playing in our own complete bridge carriers.
        Prioritizes defending threatened bridges if any.
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]

        # Determine current player's piece type
        current_player = state.get_next_player()
        piece_type = current_player.get_piece_type()

        # Quick check: do we need to defend a threatened bridge?
        if piece_type == self.piece_type:
            urgent = self.find_urgent_bridge_responses(state)
            if urgent:
                urgent_pos = urgent[0][0]  # Most urgent response
                for action in actions:
                    next_state = action.get_next_game_state()
                    move_pos = self._get_move_position(state, next_state)
                    if move_pos == urgent_pos:
                        return action  # Defend the bridge!

            # Get positions to avoid (our complete bridge carriers)
            avoid_positions = self.find_own_complete_bridge_carriers(state)
        else:
            avoid_positions = set()

        best_action = None
        best_bridge_score = -1

        # Sample a subset of actions for speed
        sample_size = min(15, len(actions))
        sampled = random.sample(actions, sample_size)

        for action in sampled:
            next_state = action.get_next_game_state()
            move_pos = self._get_move_position(state, next_state)

            # Skip moves in our own complete bridge carriers (wasteful!)
            if move_pos in avoid_positions:
                continue

            # Quick check: does this move form a diagonal bridge?
            score = 0
            for di, dj in [(1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1)]:
                partner = (move_pos[0] + di, move_pos[1] + dj)
                if partner in env and env[partner].get_type() == piece_type:
                    score += 1

            if score > best_bridge_score:
                best_bridge_score = score
                best_action = action

        return best_action if best_action else random.choice(actions)

    # ========== ROOT MOVE ORDERING ==========

    def ordered_root_actions(self, state: GameState) -> List:
        """
        Order root moves by quality using REACTIVE bridge strategy.

        Priority order:
        1. URGENT: Defend our threatened bridges (MUST respond or lose connection)
        2. Immediate wins
        3. Critical bridge cuts (block opponent's dangerous bridges)
        4. Bridge-forming moves (but NOT in our own complete bridge carriers)
        5. Other moves by accurate heuristic
        6. DEPRIORITIZED: Playing in our own complete bridge carriers (waste of move)

        Key insight: Complete bridges are virtual connections. Don't fill them
        unless opponent intrudes. If opponent plays one carrier, MUST play the other.
        
        BALANCED STRATEGY: When behind on path distance, prioritize blocking more.
        When ahead, can focus on bridge-building.
        """
        actions = list(state.get_possible_heavy_actions())
        if not actions:
            return actions

        env = state.get_rep().get_env()

        # Calculate path advantage to determine blocking vs bridging balance
        my_dist = self.shortest_path_distance(state, self.piece_type, consider_bridges=True)
        opp_dist = self.shortest_path_distance(state, self.opp_type, consider_bridges=True)
        path_deficit = my_dist - opp_dist  # Positive = opponent is closer to winning
        
        # Dynamic bonuses based on position
        # When behind: boost blocking bonus, reduce bridging bonus
        # When ahead: can afford to build bridges
        if path_deficit > 2:
            # We're significantly behind - MUST block!
            cut_bonus = self.BONUS_CUT * (1.0 + path_deficit * 0.2)  # Scale up
            bridge_bonus = self.BONUS_BRIDGE * 0.5  # Scale down
        elif path_deficit > 0:
            # Slightly behind - balanced with slight blocking preference
            cut_bonus = self.BONUS_CUT * 1.2
            bridge_bonus = self.BONUS_BRIDGE * 0.8
        else:
            # Even or ahead - can focus more on bridges
            cut_bonus = self.BONUS_CUT
            bridge_bonus = self.BONUS_BRIDGE

        # Pre-compute strategic move sets
        urgent_responses = {pos for pos, _ in self.find_urgent_bridge_responses(state)}
        own_complete_carriers = self.find_own_complete_bridge_carriers(state)
        
        # Combine bridge cuts and path cuts for comprehensive blocking
        bridge_cuts_list = self.find_cut_points(state, self.opp_type)[:5]
        path_cuts_list = self.find_path_cut_points(state, self.opp_type)[:10]
        
        # ENHANCED: Merge both types of cuts, combining priorities for overlapping cells
        # The enhanced methods now include vital points, corridor factors, connectivity, and ladder breaks
        cut_priorities = {}
        cut_metadata = {}  # Store breakdown of scores for debug
        
        for pos, priority in bridge_cuts_list:
            cut_priorities[pos] = cut_priorities.get(pos, 0) + priority
            cut_metadata[pos] = {'bridge_cut': priority}
            
        for pos, priority in path_cuts_list:
            # Path cuts weighted slightly less than bridge cuts
            cut_priorities[pos] = cut_priorities.get(pos, 0) + priority * 0.8
            if pos not in cut_metadata:
                cut_metadata[pos] = {}
            cut_metadata[pos]['path_cut'] = priority * 0.8
        
        cut_points = set(cut_priorities.keys())
        
        # Identify vital points (high-priority cuts that block multiple paths)
        vital_points = {pos for pos, priority in cut_priorities.items() if priority > 15.0}
        
        bridge_opps = {pos for pos, _ in self.find_bridge_opportunities(state, self.piece_type)[:10]}

        # Remove our own complete carriers from bridge opportunities
        # (don't form bridges by filling existing bridge carriers)
        bridge_opps = bridge_opps - own_complete_carriers

        # Categorize moves - ENHANCED with vital points category
        urgent_defense = []  # MUST play - defend threatened bridges
        wins = []
        vital_cuts = []  # NEW: Critical vital point cuts (between urgent and regular cuts)
        critical_cuts = []
        bridge_moves = []
        others = []
        wasteful = []  # Playing in own complete bridge carriers - avoid!

        for action in actions:
            next_state = action.get_next_game_state()

            # Find the move position (difference between states)
            move_pos = self._get_move_position(state, next_state)

            if move_pos in urgent_responses:
                # HIGHEST PRIORITY: Defend our threatened bridge
                value = self.W_BRIDGE_DEFENSE  # Very high priority
                urgent_defense.append((action, value))

            elif next_state.is_done() and next_state.get_player_score(self) > 0:
                # Category 2: Immediate win
                wins.append((action, float('inf')))

            elif move_pos in own_complete_carriers:
                # LOWEST PRIORITY: Wasting a move filling our own complete bridge
                # The bridge is already a virtual connection!
                value = self.accurate_heuristic(next_state) - 50.0  # Heavy penalty
                wasteful.append((action, value))

            elif move_pos in vital_points:
                # Category 3: VITAL POINT - High-priority cut blocking multiple paths
                value = self.accurate_heuristic(next_state)
                # Vital points get extra bonus due to their strategic importance
                priority = cut_priorities.get(move_pos, 1.0)
                value += cut_bonus * (priority / 4.0) * 1.3  # 30% boost for vital points
                vital_cuts.append((action, value))
                
            elif move_pos in cut_points:
                # Category 4: Critical bridge/path cut - BALANCED with path deficit
                value = self.accurate_heuristic(next_state)
                # Add dynamic bonus scaled by cut priority
                priority = cut_priorities.get(move_pos, 1.0)
                value += cut_bonus * (priority / 4.0)  # Normalize priority contribution
                critical_cuts.append((action, value))

            elif move_pos in bridge_opps:
                # Category 5: Bridge-forming move
                value = self.accurate_heuristic(next_state)
                value += bridge_bonus
                bridge_moves.append((action, value))

            else:
                # Category 6: Other moves
                value = self.accurate_heuristic(next_state)
                others.append((action, value))

        # Sort each category by value
        urgent_defense.sort(key=lambda x: -x[1])
        vital_cuts.sort(key=lambda x: -x[1])
        critical_cuts.sort(key=lambda x: -x[1])
        bridge_moves.sort(key=lambda x: -x[1])
        others.sort(key=lambda x: -x[1])
        wasteful.sort(key=lambda x: -x[1])

        # ENHANCED Priority: urgent defense > wins > vital cuts > cuts > bridges > others > wasteful
        all_moves = urgent_defense + wins + vital_cuts + critical_cuts + bridge_moves + others + wasteful

        # ENHANCED Debug: Show top moves with enhanced blocking info
        if len(all_moves) > 0:
            top_action, top_value = all_moves[0]
            top_pos = self._get_move_position(state, top_action.get_next_game_state())
            category = "URGENT_DEFENSE" if top_action in [a for a, _ in urgent_defense] else \
                      "WIN" if top_action in [a for a, _ in wins] else \
                      "VITAL_CUT" if top_action in [a for a, _ in vital_cuts] else \
                      "CUT" if top_action in [a for a, _ in critical_cuts] else \
                      "BRIDGE" if top_action in [a for a, _ in bridge_moves] else \
                      "OTHER" if top_action in [a for a, _ in others] else "WASTEFUL"
            
            # Show metadata if it's a cut move
            metadata_str = ""
            if top_pos in cut_metadata:
                meta = cut_metadata[top_pos]
                parts = []
                if 'bridge_cut' in meta:
                    parts.append(f"bridge:{meta['bridge_cut']:.1f}")
                if 'path_cut' in meta:
                    parts.append(f"path:{meta['path_cut']:.1f}")
                if parts:
                    metadata_str = f" [{', '.join(parts)}]"
            
            print(f"🎲 Top move: {top_pos} [{category}] (value: {top_value:.2f}){metadata_str}")
            
            # Show balance info with vital cuts
            print(f"📊 Move distribution: {len(urgent_defense)} urgent, {len(wins)} wins, "
                  f"{len(vital_cuts)} vital, {len(critical_cuts)} cuts, {len(bridge_moves)} bridges, "
                  f"{len(others)} others, {len(wasteful)} wasteful")

        return [action for action, _ in all_moves]

    def _get_move_position(self, state: GameState, next_state: GameState) -> Tuple[int, int]:
        """Find the position where a piece was placed between two states."""
        current_env = state.get_rep().get_env()
        next_env = next_state.get_rep().get_env()

        for pos in next_env:
            if pos not in current_env:
                return pos

        # Fallback (shouldn't happen)
        return (-1, -1)

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

    # ========== TREE-BASED MCTS METHODS ==========
    
    def mcts_select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse tree using UCB1 until reaching a node that
        can be expanded or is terminal.
        
        Args:
            node: Starting node (typically root)
            
        Returns:
            Node to expand or evaluate
        """
        current = node
        
        while not current.is_terminal:
            if not current.is_fully_expanded():
                # Found a node with unexpanded actions
                return current
            else:
                # All actions expanded, select best child by UCB
                current = current.best_child(c=1.4)
        
        return current
    
    def mcts_expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add one child node for an unexpanded action.
        
        Uses move ordering heuristic to prioritize which action to expand first.
        
        Args:
            node: Node to expand
            
        Returns:
            Newly created child node
        """
        if node.is_terminal:
            return node
        
        unexpanded = node.get_unexpanded_actions()
        if not unexpanded:
            return node
        
        # Order unexpanded actions by heuristic if we have many
        # For efficiency, only order if there are many unexpanded actions
        if len(unexpanded) > 10 and node.player_ref:
            # Quick heuristic scoring for action ordering
            action_scores = []
            for action in unexpanded:
                next_state = action.get_next_game_state()
                score = node.player_ref.accurate_heuristic(next_state)
                action_scores.append((action, score))
            
            # Sort by score (descending) and take best
            action_scores.sort(key=lambda x: -x[1])
            action = action_scores[0][0]
        else:
            # Just take first unexpanded action
            action = unexpanded[0]
        
        # Remove from unexpanded list
        node.get_unexpanded_actions().remove(action)
        
        # Create child node
        next_state = action.get_next_game_state()
        child = MCTSNode(next_state, parent=node, action=action, player_ref=node.player_ref)
        node.children[action] = child
        
        return child
    
    def mcts_simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: Evaluate the position.
        
        For efficiency, we use accurate_heuristic instead of full rollout.
        This is faster and often more accurate for tactical positions.
        
        Args:
            node: Node to evaluate
            
        Returns:
            Evaluation score from current player's perspective
        """
        if node.player_ref is None:
            # Fallback: simple random rollout
            return self.biased_rollout(node.state)
        
        # Check if terminal
        if node.is_terminal:
            if node.state.get_player_score(node.player_ref) > 0:
                return 1000.0  # We won
            else:
                return -1000.0  # We lost
        
        # Use accurate heuristic for leaf evaluation
        score = node.player_ref.accurate_heuristic(node.state)
        
        # Normalize score to reasonable range (-100 to 100 typical)
        # This helps with value backpropagation
        return score
    
    def mcts_backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagation phase: Update statistics for all nodes on path to root.
        
        Args:
            node: Leaf node where simulation ended
            value: Simulation result value
        """
        current = node
        
        while current is not None:
            current.update(value)
            current = current.parent
            
            # Flip value for opponent (alternating players)
            # This assumes perfect alternation between players
            value = -value

    # ========== MAIN ENTRY POINT ==========

    def compute_action(
            self,
            current_state: GameState,
            remaining_time: int = int(1e9),
            **kwargs
    ) -> Action:
        """
        Main decision function: Tree-based Monte Carlo Tree Search (MCTS).

        Algorithm:
        1. Create root node
        2. Allocate time for this move
        3. Run MCTS iterations until time runs out:
           - Selection: Navigate tree using UCB1
           - Expansion: Add one new child node
           - Simulation: Evaluate position (using heuristic)
           - Backpropagation: Update all nodes on path
        4. Return action leading to most visited child (robust selection)
        
        Tree-based MCTS provides 2-4 move lookahead naturally, as promising
        branches get expanded deeper through UCB-guided selection.
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

        # Print bridge analysis for this position
        self.print_bridge_analysis(current_state)

        # Check if only one legal move
        actions = list(current_state.get_possible_heavy_actions())
        if not actions:
            raise MethodNotImplementedError("No legal actions available")
        
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

        # Create root node with player reference for heuristic evaluation
        root = MCTSNode(current_state, parent=None, action=None, player_ref=self)
        
        # MCTS main loop
        iterations = 0
        while time.time() < deadline:
            # 1. Selection: traverse tree to find node to expand
            node = self.mcts_select(root)
            
            # 2. Expansion: add one child if not terminal
            if not node.is_terminal and not node.is_fully_expanded():
                node = self.mcts_expand(node)
            
            # 3. Simulation: evaluate the position
            value = self.mcts_simulate(node)
            
            # 4. Backpropagation: update statistics
            self.mcts_backpropagate(node, value)
            
            iterations += 1
        
        # Select best action: most visited child (robust child selection)
        if not root.children:
            # Fallback: no children expanded (time ran out immediately)
            # Use heuristic-ordered actions
            ordered_actions = self.ordered_root_actions(current_state)
            return current_state.convert_heavy_action_to_light_action(ordered_actions[0])
        
        best_action, best_child = root.most_visited_child()
        
        # Debug output
        print(f"🌲 Tree MCTS: {iterations} iterations, {len(root.children)} root children")
        print(f"   Best move visits: {best_child.visits}, avg value: {best_child.total_value/max(1, best_child.visits):.2f}")
        
        # Find max depth reached in tree
        max_depth = self._get_tree_depth(root)
        print(f"   Tree depth reached: {max_depth}")
        
        return current_state.convert_heavy_action_to_light_action(best_action)
    
    def _get_tree_depth(self, node: MCTSNode, current_depth: int = 0) -> int:
        """Helper to calculate maximum depth of tree."""
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth

    # ========== LADDER DETECTION (Advanced) ==========

    def detect_ladder(self, state: GameState, piece_type: str,
                      start_pos: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Detect if there's a forcing ladder sequence from a given position.

        A ladder in Hex is a forcing sequence where each move forces the opponent
        to respond, eventually reaching the goal edge.

        Returns:
            List of positions forming the ladder, or None if no ladder exists.
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        opp_type = "B" if piece_type == "R" else "R"

        # Simplified ladder detection: check if there's a forcing path
        # This is a basic implementation - full ladder detection is complex

        visited = set()
        ladder_path = []

        def can_force(pos: Tuple[int, int], depth: int) -> bool:
            """Check if we can force a connection from this position."""
            if depth > 15:  # Limit search depth
                return False

            i, j = pos
            visited.add(pos)
            ladder_path.append(pos)

            # Check if reached goal
            if piece_type == "R" and i == dim - 1:
                return True
            if piece_type == "B" and j == dim - 1:
                return True

            # Find forcing moves (moves that the opponent must respond to)
            neighbors = state.get_neighbours(i, j)
            forcing_options = []

            for direction, (ntype, (ni, nj)) in neighbors.items():
                if (ni, nj) in visited:
                    continue

                if ntype == "OUTSIDE":
                    continue
                elif ntype == piece_type:
                    # Already have a piece here - can continue
                    forcing_options.append(((ni, nj), 0))  # Free
                elif ntype == "EMPTY":
                    # Check if playing here forces opponent response
                    if self._is_forcing_move((ni, nj), piece_type, env, dim):
                        forcing_options.append(((ni, nj), 1))

            # Try forcing options in order of cost
            forcing_options.sort(key=lambda x: x[1])

            for (next_pos, cost) in forcing_options:
                if can_force(next_pos, depth + cost):
                    return True

            ladder_path.pop()
            return False

        if can_force(start_pos, 0):
            return ladder_path
        return None

    def detect_ladder_break_for_block(self, state: GameState, pos: Tuple[int, int],
                                      piece_type: str, opponent_type: str) -> float:
        """
        Enhanced: Detect if blocking at pos breaks opponent's ladder/forcing sequence.
        
        This is specifically for blocking evaluation - checks if our block
        disrupts opponent's forcing path to goal.
        
        Args:
            state: Current game state
            pos: Position where we consider blocking
            piece_type: Our piece type
            opponent_type: Opponent's piece type
            
        Returns:
            Ladder break value (0.0 if no ladder broken, higher = more important break)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        if pos in env:
            return 0.0
        
        # Check if opponent has a ladder through this position
        # Look for opponent stones adjacent to pos that form forcing sequence
        neighbors = state.get_neighbours(pos[0], pos[1])
        opp_neighbors = []
        
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == opponent_type:
                opp_neighbors.append((ni, nj))
        
        if len(opp_neighbors) < 2:
            return 0.0  # Not breaking a connection
        
        ladder_value = 0.0
        
        # Check if opponent neighbors form a chain toward goal
        if opponent_type == "R":
            # Check if opp pieces are advancing downward
            opp_rows = [ni for ni, nj in opp_neighbors]
            avg_row = sum(opp_rows) / len(opp_rows)
            progress = avg_row / (dim - 1)
            
            # Higher value if they're closer to goal
            if progress > 0.5:
                ladder_value += 3.0 * progress
                
            # Check if pieces form vertical chain
            if max(opp_rows) - min(opp_rows) >= 2:
                ladder_value += 2.0  # Breaking vertical ladder
        else:  # Blue
            opp_cols = [nj for ni, nj in opp_neighbors]
            avg_col = sum(opp_cols) / len(opp_cols)
            progress = avg_col / (dim - 1)
            
            if progress > 0.5:
                ladder_value += 3.0 * progress
                
            if max(opp_cols) - min(opp_cols) >= 2:
                ladder_value += 2.0  # Breaking horizontal ladder
        
        # Additional check: are the opponent pieces connected?
        # If blocking here separates them, that's valuable
        opp_set = set(opp_neighbors)
        for opp_pos in opp_neighbors:
            opp_neighs = state.get_neighbours(opp_pos[0], opp_pos[1])
            connected_count = 0
            for _, (n_type, n_pos) in opp_neighs.values():
                if n_type == opponent_type and n_pos in opp_set and n_pos != opp_pos:
                    connected_count += 1
            
            if connected_count == 0:
                # This opponent piece only connects via pos - critical!
                ladder_value += 2.0
        
        return ladder_value

    def _is_forcing_move(self, pos: Tuple[int, int], piece_type: str,
                         env: Dict, dim: int) -> bool:
        """
        Check if playing at pos creates a forcing threat.

        A move is forcing if it creates multiple threats that opponent
        cannot all block in one move.
        """
        i, j = pos

        # Count how many goal-direction threats this creates
        threat_count = 0

        # Check adjacent cells that would create bridge opportunities
        offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < dim and 0 <= nj < dim:
                if (ni, nj) in env and env[(ni, nj)].get_type() == piece_type:
                    threat_count += 1

        # Also check goal proximity
        if piece_type == "R":
            goal_dist = dim - 1 - i
        else:
            goal_dist = dim - 1 - j

        # Consider it forcing if it has connections and is close to goal
        return threat_count >= 1 and goal_dist <= 4

    # ========== DEBUGGING & ANALYSIS ==========

    def analyze_position(self, state: GameState) -> Dict:
        """
        Comprehensive position analysis for debugging.

        Returns dictionary with:
        - path_distances: shortest path for each player
        - bridges: bridge structures for each player (complete, secured, threatened)
        - urgent_responses: positions we MUST play to defend threatened bridges
        - avoid_positions: our complete bridge carriers (don't waste moves here!)
        - cut_points: critical blocking points
        - bridge_opportunities: potential bridge-forming moves
        """
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()

        # Path distances (with virtual connections)
        my_path = self.shortest_path_distance(state, self.piece_type, consider_bridges=False)
        my_virtual = self.virtual_connection_distance(state, self.piece_type)
        opp_path = self.shortest_path_distance(state, self.opp_type, consider_bridges=False)
        opp_virtual = self.virtual_connection_distance(state, self.opp_type)

        # Bridge analysis
        my_bridges = self.find_bridges(state, self.piece_type)
        opp_bridges = self.find_bridges(state, self.opp_type)

        # Threatened bridges and urgent responses
        threatened = self.find_threatened_bridges(state)
        urgent_responses = self.find_urgent_bridge_responses(state)

        # Positions to avoid (our complete bridge carriers)
        avoid_positions = self.find_own_complete_bridge_carriers(state)

        # Cut points and opportunities
        cut_points = self.find_cut_points(state, self.opp_type)
        bridge_opps = self.find_bridge_opportunities(state, self.piece_type)

        # Remove avoid positions from opportunities
        bridge_opps = [(pos, val) for pos, val in bridge_opps if pos not in avoid_positions]

        # Heuristic components
        center = (dim - 1) / 2.0
        my_center = sum(1.0 / (1.0 + abs(i - center) + abs(j - center))
                       for (i, j), p in env.items() if p.get_type() == self.piece_type)
        opp_center = sum(1.0 / (1.0 + abs(i - center) + abs(j - center))
                        for (i, j), p in env.items() if p.get_type() == self.opp_type)

        # Dynamic center weight and game progress
        dynamic_center_weight = self.get_dynamic_center_weight(state)
        game_phase = self.get_game_phase(state)
        game_progress = state.get_step() / state.max_step

        return {
            'path_distances': {
                'my_raw': my_path,
                'my_virtual': my_virtual,
                'opp_raw': opp_path,
                'opp_virtual': opp_virtual,
            },
            'bridges': {
                'my_count': len(my_bridges),
                'my_complete': sum(1 for b in my_bridges if b['complete']),
                'my_secured': sum(1 for b in my_bridges if b['secured']),
                'my_threatened': len(threatened),
                'opp_count': len(opp_bridges),
                'opp_complete': sum(1 for b in opp_bridges if b['complete']),
            },
            'urgent_responses': urgent_responses,
            'avoid_positions': avoid_positions,
            'cut_points': cut_points[:5],
            'bridge_opportunities': bridge_opps[:5],
            'center_control': {
                'my_score': my_center,
                'opp_score': opp_center,
                'dynamic_weight': dynamic_center_weight,
            },
            'game_phase': game_phase,
            'game_progress': game_progress,
            'heuristic_value': self.accurate_heuristic(state),
        }

    def print_bridge_analysis(self, state: GameState) -> None:
        """Print human-readable bridge analysis."""
        analysis = self.analyze_position(state)
        
        goal_desc = "TOP(row 0)→BOTTOM(row N-1)" if self.piece_type == "R" else "LEFT(col 0)→RIGHT(col N-1)"

        print(f"\n{'='*50}")
        print(f"POSITION ANALYSIS - {self.piece_type} to move")
        print(f"🎯 Goal: {goal_desc}")
        print(f"{'='*50}")

        print(f"\n📏 PATH DISTANCES:")
        print(f"   My path (raw):     {analysis['path_distances']['my_raw']}")
        print(f"   My path (virtual): {analysis['path_distances']['my_virtual']}")
        print(f"   Opp path (raw):    {analysis['path_distances']['opp_raw']}")
        print(f"   Opp path (virtual):{analysis['path_distances']['opp_virtual']}")

        print(f"\n🌉 BRIDGES:")
        print(f"   My bridges:  {analysis['bridges']['my_count']} total")
        print(f"      Complete: {analysis['bridges']['my_complete']} (virtual connections - don't fill!)")
        print(f"      Secured:  {analysis['bridges']['my_secured']}")
        print(f"      Threatened: {analysis['bridges']['my_threatened']} (MUST DEFEND!)")
        print(f"   Opp bridges: {analysis['bridges']['opp_count']} total")
        print(f"      Complete: {analysis['bridges']['opp_complete']}")

        # Show urgent responses
        if analysis['urgent_responses']:
            print(f"\n🚨 URGENT - MUST DEFEND BRIDGES:")
            for pos, urgency in analysis['urgent_responses']:
                print(f"   PLAY {pos} (urgency: {urgency:.2f})")

        # Show positions to avoid
        if analysis['avoid_positions']:
            print(f"\n⛔ AVOID (own complete bridge carriers):")
            for pos in list(analysis['avoid_positions'])[:5]:
                print(f"   {pos} - don't waste a move here!")

        print(f"\n⚔️  CUT POINTS (top 5):")
        for pos, priority in analysis['cut_points']:
            print(f"   {pos} (priority: {priority:.2f})")

        print(f"\n🎯 BRIDGE OPPORTUNITIES (top 5):")
        for pos, value in analysis['bridge_opportunities']:
            print(f"   {pos} (value: {value:.2f})")

        print(f"\n🎲 CENTER CONTROL:")
        print(f"   My score:  {analysis['center_control']['my_score']:.2f}")
        print(f"   Opp score: {analysis['center_control']['opp_score']:.2f}")
        print(f"   Current weight: {analysis['center_control']['dynamic_weight']:.2f}")
        print(f"   (decays from {self.W_CENTER_BASE} to {self.W_CENTER_BASE * self.W_CENTER_DECAY:.2f})")

        print(f"\n🎮 GAME PHASE: {analysis['game_phase'].upper()}")
        print(f"   Progress: {analysis['game_progress']*100:.1f}%")

        print(f"\n📊 HEURISTIC VALUE: {analysis['heuristic_value']:.2f}")
        
        # Show blocking/bridging balance
        path_deficit = analysis['path_distances']['my_virtual'] - analysis['path_distances']['opp_virtual']
        if path_deficit > 2:
            strategy = "DEFENSIVE (focus on blocking!)"
            cut_mult = 1.0 + path_deficit * 0.2
            bridge_mult = 0.5
        elif path_deficit > 0:
            strategy = "BALANCED (slight blocking preference)"
            cut_mult = 1.2
            bridge_mult = 0.8
        else:
            strategy = "OFFENSIVE (can build bridges)"
            cut_mult = 1.0
            bridge_mult = 1.0
        
        print(f"\n⚖️  STRATEGY BALANCE:")
        print(f"   Path deficit: {path_deficit:+.0f} (positive = behind)")
        print(f"   Strategy: {strategy}")
        print(f"   Cut bonus multiplier:    {cut_mult:.1f}x")
        print(f"   Bridge bonus multiplier: {bridge_mult:.1f}x")
        print(f"{'='*50}\n")

    # ========== WEIGHT TUNING INTERFACE ==========

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
        """
        ENHANCED: Factory method to create player with custom weights.

        Center weight decays exponentially: W_CENTER = BASE * (DECAY ^ progress)
        - At game start: W_CENTER = 5.0 (very important)
        - At game end:   W_CENTER = 5.0 * 0.3 = 1.5 (less important)
        
        Move ordering bonuses (BONUS_CUT vs BONUS_BRIDGE) control the 
        blocking vs bridge-building balance. Default ratio is 2:1 favoring blocking.
        
        Enhanced blocking weights:
        - w_vital_point: Weight for vital point detection (blocks multiple paths)
        - w_corridor: Multiplier for moves in opponent's corridor
        - w_connectivity: Weight for connectivity scoring (dual-purpose moves)
        - w_ladder_break: Weight for ladder break detection
        - overconcentration_penalty: Penalty for clustering (< 1.0 reduces score)

        Example:
            player = MyPlayer.create_with_weights(
                "R",
                w_center_base=6.0,   # Higher initial center importance
                w_center_decay=0.2,  # Faster decay
                w_bridge=5.0,
                w_vital_point=8.0,   # More aggressive vital point blocking
                w_connectivity=4.0,  # Value dual-purpose moves more
                bonus_cut=10.0,      # More aggressive blocking
                bonus_bridge=3.0     # Less focus on bridging
            )
        """
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
        """Return current heuristic weights including enhanced blocking weights."""
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
        """
        Update heuristic weights.

        Example:
            player.set_weights(W_BRIDGE=5.0, W_BLOCK=4.0)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

