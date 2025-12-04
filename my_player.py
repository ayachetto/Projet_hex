# Aya Chetto (2194547)
# Camille Ménard (2214742)

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
import math, random, time
from typing import Dict, Tuple, List, Set, Optional
import heapq
from collections import deque


class MCTSNode:
    """
    Node in the MCTS tree for deeper lookahead.
    
    Attributes prefixed with '_' are filtered from Abyss JSON output (safe).
    """
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Action] = None, player_ref = None):
        
        self.state = state
        self.parent = parent
        self._action = action # Prefixed for JSON filtering
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
        """Get list of actions not yet expanded as children (SECURED)."""
        if self._unexpanded_actions is None:
            if self.is_terminal:
                self._unexpanded_actions = []
                self._all_actions = []
            else:
                try:
                    self._all_actions = list(self.state.get_possible_heavy_actions())
                except Exception:
                    # Sécurité: Si l'accès aux actions échoue, considérer comme terminal.
                    self._all_actions = []
                    self.is_terminal = True 
                
                self._unexpanded_actions = self._all_actions.copy()
        return self._unexpanded_actions
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been expanded."""
        return len(self.get_unexpanded_actions()) == 0
    
    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        """Select best child using UCB1 formula."""
        best_value = -float('inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                ucb_value = float('inf')
            else:
                exploitation = child.total_value / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
            
        return best_child
    
    def most_visited_child(self) -> Tuple[Optional[Action], Optional['MCTSNode']]:
        """Return the child with the most visits (robust child selection)."""
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
        """Update node statistics after a simulation."""
        self.visits += 1
        self.total_value += value


class MyPlayer(PlayerHex):
    """
    Player class for Hex game with Bridge-focused MCTS + Alpha-Beta.
    """

    # ========== BRIDGE PATTERN DEFINITIONS ==========
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

    # Heuristic weights (Préfixés par _ pour éviter l'enregistrement JSON)
    _W_CENTER_BASE = 5.0
    _W_CENTER_DECAY = 0.1
    _W_PROGRESS = 2.0
    _W_BRIDGE = 4.0
    _W_BLOCK = 8.0
    _W_PATH = 10.0
    _W_BRIDGE_DEFENSE = 100.0
    
    _W_VITAL_POINT = 6.0
    _W_CORRIDOR = 1.5
    _W_CONNECTIVITY = 3.0
    _W_LADDER_BREAK = 7.0
    _OVERCONCENTRATION_PENALTY = 0.6
    
    _BONUS_CUT = 8.0
    _BONUS_BRIDGE = 4.0

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance. (SÉCURISÉ: Lazy Initialization)
        """
        super().__init__(piece_type, name)
        self.opp_type = "B" if piece_type == "R" else "R"

        self._bridge_offsets = None  
    
    def _get_bridge_offsets(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Calcule ou retourne les offsets de ponts (Calcul Paresseux)."""
        if self._bridge_offsets is None:
            self._bridge_offsets = self._precompute_bridge_offsets()
        return self._bridge_offsets

    # ========== DYNAMIC WEIGHT CALCULATION ==========

    def get_dynamic_center_weight(self, state: GameState) -> float:
        """
        Calculate center weight that decays exponentially as game progresses.
        (SECURED against ZeroDivisionError)
        """
        current_step = state.get_step()
        max_steps = state.max_step

        if max_steps == 0:
            return self._W_CENTER_BASE
        
        progress = min(1.0, max(0.0, current_step / max_steps))
        dynamic_weight = self._W_CENTER_BASE * (self._W_CENTER_DECAY ** progress)

        return dynamic_weight

    def get_game_phase(self, state: GameState) -> str:
        """
        Determine current game phase for strategic adjustments.
        (SECURED against ZeroDivisionError)
        """
        current_step = state.get_step()
        max_steps = state.max_step
            
        if max_steps == 0:
            return 'opening'
            
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
        Precompute all valid bridge offset patterns. (Logique conservée)
        """
        neighbors = [
            (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
        ]

        def neighbor_cells(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            i, j = pos
            return [(i + di, j + dj) for di, dj in neighbors]

        origin = (0, 0)
        origin_neighbors = set(neighbor_cells(origin))
        bridge_patterns = []

        for i, (d1i, d1j) in enumerate(neighbors):
            for j, (d2i, d2j) in enumerate(neighbors):
                if i >= j: continue
                partner = (d1i + d2i, d1j + d2j)
                if partner == origin or partner in origin_neighbors: continue
                partner_neighbors = set(neighbor_cells(partner))
                common = origin_neighbors & partner_neighbors
                if len(common) != 2: continue
                carrier1 = (d1i, d1j)
                carrier2 = (d2i, d2j)
                if {carrier1, carrier2} != common: continue
                bridge_patterns.append((partner, carrier1, carrier2))

        return bridge_patterns

    # ========== BRIDGE DETECTION ==========

    def find_bridges(self, state: GameState, piece_type: str) -> List[Dict]:
        """Find all bridge structures for a given player."""
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        if dim < 2: return []

        bridges = []
        seen_pairs = set()

        for (i, j), piece in env.items():
            if piece.get_type() != piece_type: continue

            for partner_off, carrier1_off, carrier2_off in self._get_bridge_offsets():
                partner = (i + partner_off[0], j + partner_off[1])
                carrier1 = (i + carrier1_off[0], j + carrier1_off[1])
                carrier2 = (i + carrier2_off[0], j + carrier2_off[1])

                if not self._in_bounds(partner, dim) or not self._in_bounds(carrier1, dim) or not self._in_bounds(carrier2, dim):
                    continue

                if partner not in env or env[partner].get_type() != piece_type: continue
                pair_key = (min((i, j), partner), max((i, j), partner))
                if pair_key in seen_pairs: continue
                seen_pairs.add(pair_key)

                c1_status = self._cell_status(carrier1, env, piece_type)
                c2_status = self._cell_status(carrier2, env, piece_type)

                complete = (c1_status == "empty" and c2_status == "empty")
                threatened = (c1_status == "opponent" or c2_status == "opponent")
                secured = (c1_status == "friendly" or c2_status == "friendly")

                if c1_status != "opponent" or c2_status != "opponent":
                    bridges.append({
                        'stone1': (i, j), 'stone2': partner, 'carrier1': carrier1, 'carrier2': carrier2,
                        'c1_status': c1_status, 'c2_status': c2_status,
                        'complete': complete, 'threatened': threatened, 'secured': secured,
                    })

        return bridges
    
    def _in_bounds(self, pos: Tuple[int, int], dim: int) -> bool:
        """Check if position is within board bounds."""
        return 0 <= pos[0] < dim and 0 <= pos[1] < dim

    def _cell_status(self, pos: Tuple[int, int], env: Dict, piece_type: str) -> str:
        """Get the status of a cell: 'empty', 'friendly', or 'opponent'."""
        if pos not in env:
            return "empty"
        elif env[pos].get_type() == piece_type:
            return "friendly"
        else:
            return "opponent"

    # ========== THREATENED BRIDGE DETECTION ==========

    def find_threatened_bridges(self, state: GameState) -> List[Dict]:
        bridges = self.find_bridges(state, self.piece_type)
        threatened = []

        for bridge in bridges:
            if bridge['threatened'] and not bridge['secured']:
                if bridge['c1_status'] == 'empty':
                    response = bridge['carrier1']
                elif bridge['c2_status'] == 'empty':
                    response = bridge['carrier2']
                else:
                    continue  

                threatened.append({
                    **bridge,
                    'response': response,
                    'urgency': self._calculate_bridge_urgency(bridge, state)
                })

        threatened.sort(key=lambda x: -x['urgency'])
        return threatened

    def _calculate_bridge_urgency(self, bridge: Dict, state: GameState) -> float:
        dim = state.get_rep().get_dimensions()[0]
        s1, s2 = bridge['stone1'], bridge['stone2']
        urgency = 10.0

        if self.piece_type == "R":
            avg_row = (s1[0] + s2[0]) / 2.0
            progress = avg_row / (dim - 1) if dim > 1 else 0.0
            urgency += progress * 5.0
        else:
            avg_col = (s1[1] + s2[1]) / 2.0
            progress = avg_col / (dim - 1) if dim > 1 else 0.0
            urgency += progress * 5.0

        return urgency

    def find_urgent_bridge_responses(self, state: GameState) -> List[Tuple[Tuple[int, int], float]]:
        threatened = self.find_threatened_bridges(state)
        responses = []
        for bridge in threatened:
            responses.append((bridge['response'], bridge['urgency']))
        return responses

    def find_own_complete_bridge_carriers(self, state: GameState) -> Set[Tuple[int, int]]:
        bridges = self.find_bridges(state, self.piece_type)
        carriers_to_avoid = set()
        for bridge in bridges:
            if bridge['complete']:
                carriers_to_avoid.add(bridge['carrier1'])
                carriers_to_avoid.add(bridge['carrier2'])
        return carriers_to_avoid

    # ========== IMPLEMENTATION DES PLACEHOLDERS NÉCESSAIRES POUR AB ==========
    
    # Placeholder for remaining methods
    def score_bridges(self, state: GameState, piece_type: str) -> float: return 0.0
    def local_connectivity_score(self, state: GameState, pos: Tuple[int, int], piece_type: str) -> float: return 0.0
    def find_cut_points(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]: return []
    def detect_vital_point(self, state: GameState, pos: Tuple[int, int], piece_type: str, opponent_type: str) -> float: return 0.0
    def find_path_cut_points(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]: return []
    def is_in_opponent_corridor(self, state: GameState, pos: Tuple[int, int], opponent_type: str) -> Tuple[bool, float]: return False, 0.0
    def find_bridge_opportunities(self, state: GameState, piece_type: str) -> List[Tuple[Tuple[int, int], float]]: return []
    def detect_ladder_break_for_block(self, state: GameState, pos: Tuple[int, int], piece_type: str, opponent_type: str) -> float: return 0.0
    
    # --- IMPLEMENTATION BFS POUR SHORTEST_PATH_DISTANCE (Essentiel pour AB) ---
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Renvoie les 6 voisins d'une position (i, j)."""
        i, j = pos
        return [
            (i - 1, j), (i - 1, j + 1), (i, j - 1), 
            (i, j + 1), (i + 1, j - 1), (i + 1, j),
        ]

    def shortest_path_distance(self, state: GameState, piece_type: str, consider_bridges: bool = True) -> float:
        """
        Calcul de la distance du chemin le plus court (en nombre de coups) 
        en utilisant BFS. Plus la distance est faible, mieux c'est.
        """
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        
        # Définir les limites de départ et d'arrivée
        if piece_type == "R":
            # Départ: Rangée 0 (Top), Arrivée: Rangée dim-1 (Bottom)
            start_border_range = range(dim)
            end_border_index = dim - 1
            start_coord = lambda k: (0, k)
            is_end = lambda pos: pos[0] == end_border_index
            target_coord = 0 # Axe des lignes
        else: # piece_type == "B"
            # Départ: Colonne 0 (Left), Arrivée: Colonne dim-1 (Right)
            start_border_range = range(dim)
            end_border_index = dim - 1
            start_coord = lambda k: (k, 0)
            is_end = lambda pos: pos[1] == end_border_index
            target_coord = 1 # Axe des colonnes

        # Initialisation du BFS
        queue = deque()
        visited = {} # position -> distance
        
        # 1. Ajouter les cellules de la bordure de départ
        for k in start_border_range:
            pos = start_coord(k)
            # Les pierres amies ou les cellules vides adjacentes à la bordure de départ
            if pos not in env or env[pos].get_type() == piece_type:
                # La distance est 1 pour les cellules vides (coup nécessaire)
                # La distance est 0 pour les pierres déjà posées
                dist = 1 if pos not in env else 0
                queue.append((pos, dist))
                visited[pos] = dist

        min_distance = dim * dim # Distance maximale possible
        
        # BFS
        while queue:
            (curr_r, curr_c), dist = queue.popleft()
            
            # Vérification de l'arrivée (bordure opposée)
            if is_end((curr_r, curr_c)):
                min_distance = min(min_distance, dist)
                # Continuer car le BFS ne garantit pas toujours le plus court si 
                # plusieurs chemins peuvent être atteints par la même distance dans la file.
            
            # Si on a déjà trouvé un chemin plus court, on peut arrêter l'exploration à partir d'ici
            if dist >= min_distance:
                continue

            for next_r, next_c in self._get_neighbors((curr_r, curr_c)):
                next_pos = (next_r, next_c)
                
                if not self._in_bounds(next_pos, dim):
                    continue

                cell_status = self._cell_status(next_pos, env, piece_type)
                
                new_dist = dist
                if cell_status == "empty":
                    # Coût d'un mouvement
                    new_dist += 1
                elif cell_status == "friendly":
                    # Coût de 0 pour les propres pierres
                    new_dist += 0
                elif cell_status == "opponent":
                    # Coût très élevé pour un blocage (pour l'instant, on ignore les blocages simples)
                    continue 


                if next_pos not in visited or new_dist < visited[next_pos]:
                    visited[next_pos] = new_dist
                    queue.append((next_pos, new_dist))

        # Si le chemin le plus court trouvé est min_distance, cela signifie que nous avons besoin de 
        # min_distance coups pour gagner. Si non trouvé, on retourne une valeur très élevée.
        return min_distance if min_distance != dim * dim else float(dim * dim + 1)

    # --- ALPHA-BETA IMPLEMENTATION ---

    def evaluate_state(self, state: GameState, player_type: str) -> float:
        """
        Heuristic evaluation function for the Alpha-Beta search.
        Returns a high positive value if the state is good for player_type.
        """
        # A. Évaluation Terminale
        if state.is_done():
            # Très forte évaluation terminale
            if state.get_player_score(self) > 0: # Check if self (the player) is the winner
                return float('inf') 
            elif state.get_player_score(self) < 0: # Check if opponent won
                return -float('inf')
            else:
                return 0.0

        # B. Évaluation Non-Terminal - Basée sur la distance
        
        # Note: L'évaluation doit toujours être du point de vue du joueur root (self.piece_type).
        # Le Minimax gère l'alternance en inversant les signes.
        
        # 1. Distances de chemin le plus court
        my_dist = self.shortest_path_distance(state, self.piece_type)
        opp_type = "B" if self.piece_type == "R" else "R"
        opp_dist = self.shortest_path_distance(state, opp_type)
        
        # Heuristique de base: opp_dist - my_dist. Plus grand est meilleur pour nous.
        base_score = opp_dist - my_dist
        
        return base_score


    def minimax_alpha_beta(self, state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[float, Optional[Action]]:
        """
        Minimax algorithm with Alpha-Beta Pruning.
        Returns (best_value, best_action).
        """
        # Base Case: Terminal or Max Depth
        if depth == 0 or state.is_done():
            # L'évaluation est toujours faite du point de vue du joueur ROOT (self.piece_type)
            score = self.evaluate_state(state, self.piece_type)
            return score, None
        
        actions = list(state.get_possible_heavy_actions())
        if not actions:
            return self.evaluate_state(state, self.piece_type), None
        
        # Meilleure action pour le joueur courant (Max ou Min)
        best_action = None

        if maximizing_player:
            max_eval = -float('inf')
            
            # --- Move Ordering (très important pour AB) ---
            # Sortir les actions par une heuristique rapide pour trouver des coupures tôt.
            action_scores = []
            for action in actions:
                next_state = action.get_next_game_state()
                # Utilise evaluate_state (plus fiable que fast_heuristic)
                score = self.evaluate_state(next_state, self.piece_type)
                action_scores.append((action, score))
            
            action_scores.sort(key=lambda x: -x[1]) # Triez par meilleur score pour le Max player
            actions = [a for a, s in action_scores]
            # -----------------------------------------------

            for action in actions:
                next_state = action.get_next_game_state()
                # Recursive call: The next turn is the minimizing player's turn.
                current_eval, _ = self.minimax_alpha_beta(next_state, depth - 1, alpha, beta, False)
                
                if current_eval > max_eval:
                    max_eval = current_eval
                    best_action = action
                
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    # Beta cut-off
                    break
            return max_eval, best_action

        else: # Minimizing Player
            min_eval = float('inf')
            
            # --- Move Ordering (très important pour AB) ---
            action_scores = []
            for action in actions:
                next_state = action.get_next_game_state()
                score = self.evaluate_state(next_state, self.piece_type)
                action_scores.append((action, score))
            
            action_scores.sort(key=lambda x: x[1]) # Triez par pire score pour le Min player
            actions = [a for a, s in action_scores]
            # -----------------------------------------------

            for action in actions:
                next_state = action.get_next_game_state()
                # Recursive call: The next turn is the maximizing player's turn.
                current_eval, _ = self.minimax_alpha_beta(next_state, depth - 1, alpha, beta, True)
                
                if current_eval < min_eval:
                    min_eval = current_eval
                    best_action = action # L'action MINIMALE est choisie par le joueur MINIMAX
                
                beta = min(beta, min_eval)
                if beta <= alpha:
                    # Alpha cut-off
                    break
            return min_eval, best_action
    
    # --- MCTS/AB Heuristics (Placeholders modifiés pour utiliser evaluate_state) ---
    def accurate_heuristic(self, state: GameState) -> float: 
        """Utilise l'évaluation AB pour l'MCTS pour une meilleure simulation."""
        return self.evaluate_state(state, self.piece_type) 
        
    def fast_heuristic(self, state: GameState) -> float: 
        """Heuristique rapide pour l'expansion MCTS (peut être simplifiée par rapport à evaluate_state)."""
        return self.evaluate_state(state, self.piece_type) # Simple fallback

    def biased_rollout(self, state: GameState) -> float: 
        """Rollout biasé basé sur l'heuristique (peut être un simple retour de evaluate_state)."""
        return self.evaluate_state(state, self.piece_type) # Simple fallback
        
    def ordered_root_actions(self, state: GameState) -> List[Action]: 
        """Ordonne les actions à la racine pour MCTS (meilleure exploration initiale)."""
        actions = list(state.get_possible_heavy_actions())
        action_scores = []
        for action in actions:
            next_state = action.get_next_game_state()
            score = self.evaluate_state(next_state, self.piece_type)
            action_scores.append((action, score))
        
        action_scores.sort(key=lambda x: -x[1])
        return [a for a, s in action_scores]

    def _get_move_position(self, state: GameState, next_state: GameState) -> Tuple[int, int]:
        current_env = state.get_rep().get_env()
        next_env = next_state.get_rep().get_env()
        for pos in next_env:
            if pos not in current_env: return pos
        return (-1, -1)

    # ========== TIME MANAGEMENT ==========

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

    # ========== MCTS METHODES (SÉCURISÉES) ==========
    
    def mcts_select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.is_terminal:
            if not current.is_fully_expanded():
                return current
            else:
                # Utilisation de c=1.4 par défaut
                current = current.best_child(c=1.4) 
        return current
    
    def mcts_expand(self, node: MCTSNode) -> MCTSNode:
        if node.is_terminal: return node
        unexpanded = node.get_unexpanded_actions()
        if not unexpanded: return node
        
        # --- SÉLECTION D'ACTION POUR EXPANSION (Basée sur l'Heuristique) ---
        if len(unexpanded) > 10 and node.player_ref:
            action_scores = []
            for action in unexpanded:
                next_state = action.get_next_game_state()
                # Utilise une heuristique pour l'expansion
                score = node.player_ref.fast_heuristic(next_state) 
                action_scores.append((action, score))
            
            # Sélectionner la meilleure action selon l'heuristique
            action_scores.sort(key=lambda x: -x[1])
            action = action_scores[0][0]
        else:
            # Sinon, sélection arbitraire (le premier non-expandu)
            action = unexpanded[0]
        # ------------------------------------------------------------------
        
        node.get_unexpanded_actions().remove(action)
        next_state = action.get_next_game_state()
        child = MCTSNode(next_state, parent=node, action=action, player_ref=node.player_ref)
        node.children[action] = child
        
        return child
    
    def mcts_simulate(self, node: MCTSNode) -> float:
        if node.player_ref is None: return self.biased_rollout(node.state)
        
        if node.is_terminal:
            # Retourne une évaluation terminale forte
            return self.evaluate_state(node.state, self.piece_type)
        
        # Utilisez le rollout biaisé
        score = node.player_ref.biased_rollout(node.state)
        return score
    
    def mcts_backpropagate(self, node: MCTSNode, value: float):
        current = node
        while current is not None:
            # L'MCTS stocke la valeur du point de vue du joueur de la racine
            # La valeur doit être inversée pour le joueur MINIMAX (l'opposant)
            current.update(value)
            current = current.parent
            value = -value

    # ========== MAIN ENTRY POINT SÉCURISÉ POUR ABYSS (Stratégie Mixte) ==========

    def compute_action(
            self,
            current_state: GameState,
            remaining_time: int = int(1e9),
            **kwargs
    ) -> Action:
        """
        Main decision function: MCTS + Alpha-Beta Hybrid.
        """
        if kwargs.get("rng_seed") is not None:
            random.seed(kwargs["rng_seed"])

        actions = list(current_state.get_possible_heavy_actions())

        if current_state.is_done():
            if actions:
                return current_state.convert_heavy_action_to_light_action(actions[0])
            raise Exception("Called compute_action on a finished game with no legal moves.")

        if not actions:
             # Gère le cas où get_possible_heavy_actions retourne une liste vide de manière inattendue
             raise Exception("No legal actions available for non-terminal state.")


        # Check if only one legal move
        if len(actions) == 1:
            return current_state.convert_heavy_action_to_light_action(actions[0])

        # Compute time budget
        budget_ms = self.compute_per_move_budget(
            remaining_time,
            fraction=float(kwargs.get("per_move_fraction", 0.08)),
            min_ms=int(kwargs.get("per_move_min_ms", 150)),
            max_ms=int(kwargs.get("per_move_max_ms", 15000))
        )
        deadline = time.time() + budget_ms / 1000.0
        
        # --- NOUVELLE STRATÉGIE MIXTE : ALPHA-BETA EN FIN DE PARTIE ---
        game_phase = self.get_game_phase(current_state)
        
        # Activer Alpha-Beta pour la fin de partie si le nombre d'actions est gérable.
        # Le seuil d'actions est arbitraire et doit être ajusté pour les performances.
        AB_CUTOFF_ACTIONS = 15 
        AB_DEPTH = 5 # Profondeur maximale
        
        if game_phase == 'endgame' and len(actions) < AB_CUTOFF_ACTIONS: 
            
            # Allouer la moitié du budget à l'Alpha-Beta (approche conservative)
            ab_deadline = time.time() + (budget_ms / 2000.0) 
            
            # Exécuter Alpha-Beta
            # La recherche Alpha-Beta s'arrête à la profondeur maximale ou au timeout
            best_value, best_action = self.minimax_alpha_beta(
                state=current_state, 
                depth=AB_DEPTH, 
                alpha=-float('inf'), 
                beta=float('inf'), 
                maximizing_player=True
            )
            
            # Si Alpha-Beta a trouvé une action, la considérer comme définitive
            if best_action:
                return current_state.convert_heavy_action_to_light_action(best_action)
            
        # --- FIN STRATÉGIE MIXTE ---

        # Create root node (Début de l'MCTS si AB n'a pas été appliqué ou n'a pas trouvé d'action)
        root = MCTSNode(current_state, parent=None, action=None, player_ref=self)
        
        iterations = 0
        while time.time() < deadline:
            node = self.mcts_select(root)
            
            if not node.is_terminal and not node.is_fully_expanded():
                node = self.mcts_expand(node)
            
            value = self.mcts_simulate(node)
            
            self.mcts_backpropagate(node, value)
            
            iterations += 1
        
        # Select best action: most visited child
        if not root.children:
            # Fallback: utiliser l'ordre heuristique
            ordered_actions = self.ordered_root_actions(current_state)
            return current_state.convert_heavy_action_to_light_action(ordered_actions[0])
        
        best_action, best_child = root.most_visited_child()
        
        return current_state.convert_heavy_action_to_light_action(best_action)

