from seahorse.game.action import Action
from seahorse.game.game_state import GameState
import math
import random
import time
from typing import Dict, Tuple, List, Set, Optional
import heapq


class MCTSNode:
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[Action] = None, player_ref = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.player_ref = player_ref
        
        self.visits = 0
        self.total_value = 0.0
        
        self.children = {}
        
        self._unexpanded_actions = None
        self._all_actions = None
        
        self.is_terminal = state.is_done()
    
    def get_unexpanded_actions(self) -> List[Action]:
        if self._unexpanded_actions is None:
            if self.is_terminal:
                self._unexpanded_actions = []
                self._all_actions = []
            else:
                self._all_actions = list(self.state.get_possible_heavy_actions())
                self._unexpanded_actions = self._all_actions.copy()
        return self._unexpanded_actions
    
    def is_fully_expanded(self) -> bool:
        return len(self.get_unexpanded_actions()) == 0
    
    def best_child(self, c: float = 1.4) -> 'MCTSNode':
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
    
    def most_visited_child(self) -> Tuple[Action, 'MCTSNode']:
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
        self.visits += 1
        self.total_value += value
    
    def __repr__(self):
        return f"MCTSNode(visits={self.visits}, value={self.total_value:.2f}, children={len(self.children)})"


class HexPlayerHelper:
    
    def __init__(self, player):
        self.player = player
    
    def get_dynamic_center_weight(self, state: GameState) -> float:
        current_step = state.get_step()
        max_steps = state.max_step
        progress = min(1.0, max(0.0, current_step / max_steps))
        dynamic_weight = self.player.W_CENTER_BASE * (self.player.W_CENTER_DECAY ** progress)
        return dynamic_weight
    
    def get_game_phase(self, state: GameState) -> str:
        current_step = state.get_step()
        max_steps = state.max_step
        progress = current_step / max_steps
        
        if progress < 0.15:
            return 'opening'
        elif progress < 0.50:
            return 'midgame'
        else:
            return 'endgame'
    
    def _in_bounds(self, pos: Tuple[int, int], dim: int) -> bool:
        return 0 <= pos[0] < dim and 0 <= pos[1] < dim
    
    def _cell_status(self, pos: Tuple[int, int], env: Dict, piece_type: str) -> str:
        if pos not in env:
            return "empty"
        elif env[pos].get_type() == piece_type:
            return "friendly"
        else:
            return "opponent"
    
    def find_bridges(self, state: GameState, piece_type: str) -> List[Dict]:
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        opp_type = "B" if piece_type == "R" else "R"
        
        bridges = []
        seen_pairs = set()
        
        for (i, j), piece in env.items():
            if piece.get_type() != piece_type:
                continue
            
            for partner_off, carrier1_off, carrier2_off in self.player._bridge_offsets:
                partner = (i + partner_off[0], j + partner_off[1])
                carrier1 = (i + carrier1_off[0], j + carrier1_off[1])
                carrier2 = (i + carrier2_off[0], j + carrier2_off[1])
                
                if not self._in_bounds(partner, dim) or \
                   not self._in_bounds(carrier1, dim) or \
                   not self._in_bounds(carrier2, dim):
                    continue
                
                if partner not in env or env[partner].get_type() != piece_type:
                    continue
                
                pair_key = (min((i, j), partner), max((i, j), partner))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                c1_status = self._cell_status(carrier1, env, piece_type)
                c2_status = self._cell_status(carrier2, env, piece_type)
                
                complete = (c1_status == "empty" and c2_status == "empty")
                threatened = (c1_status == "opponent" or c2_status == "opponent")
                secured = (c1_status == "friendly" or c2_status == "friendly")
                
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
    
    def find_threatened_bridges(self, state: GameState) -> List[Dict]:
        bridges = self.find_bridges(state, self.player.piece_type)
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
        
        if self.player.piece_type == "R":
            avg_row = (s1[0] + s2[0]) / 2.0
            progress = avg_row / (dim - 1)
            urgency += progress * 5.0
        else:
            avg_col = (s1[1] + s2[1]) / 2.0
            progress = avg_col / (dim - 1)
            urgency += progress * 5.0
        
        return urgency
    
    def find_urgent_bridge_responses(self, state: GameState) -> List[Tuple[Tuple[int, int], float]]:
        threatened = self.find_threatened_bridges(state)
        responses = []
        
        for bridge in threatened:
            responses.append((bridge['response'], bridge['urgency']))
        
        return responses
    
    def find_own_complete_bridge_carriers(self, state: GameState) -> Set[Tuple[int, int]]:
        bridges = self.find_bridges(state, self.player.piece_type)
        carriers_to_avoid = set()
        
        for bridge in bridges:
            if bridge['complete']:
                carriers_to_avoid.add(bridge['carrier1'])
                carriers_to_avoid.add(bridge['carrier2'])
        
        return carriers_to_avoid
    
    def score_bridges(self, state: GameState, piece_type: str) -> float:
        bridges = self.find_bridges(state, piece_type)
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0
        
        total_score = 0.0
        
        for bridge in bridges:
            score = 0.0
            
            if bridge['complete']:
                score += 4.0
            elif bridge['secured']:
                score += 3.0
            elif bridge['threatened']:
                score += 0.0
            
            s1, s2 = bridge['stone1'], bridge['stone2']
            row_span = abs(s1[0] - s2[0])
            col_span = abs(s1[1] - s2[1])
            
            if piece_type == "R":
                if row_span > col_span:
                    directional_multiplier = 1.5
                elif row_span < col_span:
                    directional_multiplier = 0.7
                else:
                    directional_multiplier = 1.0
            else:
                if col_span > row_span:
                    directional_multiplier = 1.5
                elif col_span < row_span:
                    directional_multiplier = 0.7
                else:
                    directional_multiplier = 1.0
            
            score *= directional_multiplier
            
            if piece_type == "R":
                avg_row = (s1[0] + s2[0]) / 2.0
                progress = avg_row / (dim - 1)
                score += progress * 0.5 + row_span * 0.3
            else:
                avg_col = (s1[1] + s2[1]) / 2.0
                progress = avg_col / (dim - 1)
                score += progress * 0.5 + col_span * 0.3
            
            center_dist = (abs(s1[0] - center) + abs(s1[1] - center) +
                           abs(s2[0] - center) + abs(s2[1] - center)) / 2.0
            
            dynamic_center = self.get_dynamic_center_weight(state)
            center_bonus = (dynamic_center / self.player.W_CENTER_BASE) * 2.0 / (1.0 + center_dist * 0.1)
            score += center_bonus * 0.5
            
            total_score += score
        
        return total_score
    
    def local_connectivity_score(self, state: GameState, pos: Tuple[int, int], 
                                  piece_type: str) -> float:
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        i, j = pos
        
        if pos in env:
            return 0.0
        
        score = 0.0
        
        neighbors = state.get_neighbours(i, j)
        friendly_neighbors = []
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == piece_type:
                friendly_neighbors.append((ni, nj))
                score += 2.0
        
        if len(friendly_neighbors) >= 2:
            score += 3.0
        
        bridge_potential = 0.0
        for partner_off, c1_off, c2_off in self.player._bridge_offsets:
            partner = (i + partner_off[0], j + partner_off[1])
            carrier1 = (i + c1_off[0], j + c1_off[1])
            carrier2 = (i + c2_off[0], j + c2_off[1])
            
            if not self._in_bounds(partner, dim):
                continue
            
            if partner in env and env[partner].get_type() == piece_type:
                if self._in_bounds(carrier1, dim) and self._in_bounds(carrier2, dim):
                    c1_empty = carrier1 not in env
                    c2_empty = carrier2 not in env
                    
                    if c1_empty and c2_empty:
                        bridge_potential += 1.5
                    elif c1_empty or c2_empty:
                        bridge_potential += 0.8
        
        score += bridge_potential
        
        if piece_type == "R":
            has_above = any(ni < i for ni, nj in friendly_neighbors)
            has_below = any(ni > i for ni, nj in friendly_neighbors)
            if has_above and has_below:
                score += 2.0
        else:
            has_left = any(nj < j for ni, nj in friendly_neighbors)
            has_right = any(nj > j for ni, nj in friendly_neighbors)
            if has_left and has_right:
                score += 2.0
        
        return score
    
    def find_cut_points(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]:
        bridges = self.find_bridges(state, opponent_type)
        dim = state.get_rep().get_dimensions()[0]
        piece_type = "B" if opponent_type == "R" else "R"
        
        cut_points = {}
        
        for bridge in bridges:
            if not bridge['complete']:
                continue
            
            c1, c2 = bridge['carrier1'], bridge['carrier2']
            s1, s2 = bridge['stone1'], bridge['stone2']
            
            base_priority = 1.0
            
            if opponent_type == "R":
                avg_row = (s1[0] + s2[0]) / 2.0
                progress = avg_row / (dim - 1)
                base_priority += progress * 3.0
            else:
                avg_col = (s1[1] + s2[1]) / 2.0
                progress = avg_col / (dim - 1)
                base_priority += progress * 3.0
            
            for carrier_pos in [c1, c2]:
                priority = base_priority
                
                vital_score = self.detect_vital_point(state, carrier_pos, piece_type, opponent_type)
                priority += vital_score * self.player.W_VITAL_POINT / 10.0
                
                in_corridor, corridor_strength = self.is_in_opponent_corridor(state, carrier_pos, opponent_type)
                if in_corridor:
                    corridor_multiplier = 1.0 + (corridor_strength * (self.player.W_CORRIDOR - 1.0))
                    priority *= corridor_multiplier
                else:
                    priority *= 0.7
                
                connectivity = self.local_connectivity_score(state, carrier_pos, piece_type)
                priority += connectivity * self.player.W_CONNECTIVITY / 10.0
                
                ladder_break_value = self.detect_ladder_break_for_block(
                    state, carrier_pos, piece_type, opponent_type
                )
                priority += ladder_break_value * self.player.W_LADDER_BREAK / 10.0
                
                if carrier_pos not in cut_points or priority > cut_points[carrier_pos]:
                    cut_points[carrier_pos] = priority
        
        result = [(pos, priority) for pos, priority in cut_points.items()]
        result.sort(key=lambda x: -x[1])
        
        return result
    
    def detect_vital_point(self, state: GameState, pos: Tuple[int, int], 
                          piece_type: str, opponent_type: str) -> float:
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        if pos in env:
            return 0.0
        
        current_dist = self.shortest_path_distance(state, opponent_type, consider_bridges=False)
        
        neighbors = state.get_neighbours(pos[0], pos[1])
        opp_neighbor_count = 0
        opp_neighbors = []
        
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == opponent_type:
                opp_neighbor_count += 1
                opp_neighbors.append((ni, nj))
        
        vital_score = 0.0
        
        if opp_neighbor_count >= 2:
            vital_score += opp_neighbor_count * 2.0
        
        if opponent_type == "R":
            row_progress = pos[0] / (dim - 1)
            if 0.3 < row_progress < 0.7:
                vital_score += 2.0
        else:
            col_progress = pos[1] / (dim - 1)
            if 0.3 < col_progress < 0.7:
                vital_score += 2.0
        
        if len(opp_neighbors) >= 2:
            opp_neighbor_set = set(opp_neighbors)
            separated = True
            for opp_pos in opp_neighbors:
                opp_neighs = state.get_neighbours(opp_pos[0], opp_pos[1])
                for _, (n_type, n_pos) in opp_neighs.items():
                    if n_type == opponent_type and n_pos in opp_neighbor_set and n_pos != opp_pos:
                        separated = False
                        break
                if not separated:
                    break
            
            if separated:
                vital_score += 4.0
        
        ladder_break_value = self._check_ladder_break(state, pos, piece_type, opponent_type)
        vital_score += ladder_break_value
        
        return vital_score
    
    def _check_ladder_break(self, state: GameState, pos: Tuple[int, int],
                           piece_type: str, opponent_type: str) -> float:
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        neighbors = state.get_neighbours(pos[0], pos[1])
        
        opp_count = 0
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == opponent_type:
                opp_count += 1
        
        if opp_count >= 2:
            if opponent_type == "R":
                progress = pos[0] / (dim - 1)
                if progress > 0.4:
                    return 3.0
            else:
                progress = pos[1] / (dim - 1)
                if progress > 0.4:
                    return 3.0
        
        return 0.0
    
    def find_path_cut_points(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]:
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        
        distances = {}
        parents = {}
        pq = []
        
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
        else:
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
        
        goal_cells = set()
        min_goal_dist = float('inf')
        
        while pq:
            dist, i, j = heapq.heappop(pq)
            
            if (i, j) in distances and dist > distances[(i, j)]:
                continue
            
            is_goal = (opponent_type == "R" and i == dim - 1) or \
                     (opponent_type == "B" and j == dim - 1)
            
            if is_goal:
                if dist < min_goal_dist:
                    min_goal_dist = dist
                    goal_cells = {(i, j)}
                elif dist == min_goal_dist:
                    goal_cells.add((i, j))
                continue
            
            neighbors = state.get_neighbours(i, j)
            for neighbor_type, (ni, nj) in neighbors.values():
                if neighbor_type == "OUTSIDE":
                    continue
                
                if neighbor_type == opponent_type:
                    new_dist = dist
                elif neighbor_type == "EMPTY":
                    new_dist = dist + 1
                else:
                    continue
                
                if (ni, nj) not in distances or new_dist < distances[(ni, nj)]:
                    distances[(ni, nj)] = new_dist
                    parents[(ni, nj)] = [(i, j)]
                    heapq.heappush(pq, (new_dist, ni, nj))
                elif new_dist == distances[(ni, nj)]:
                    if (i, j) not in parents[(ni, nj)]:
                        parents[(ni, nj)].append((i, j))
        
        path_cells = set()
        path_frequency = {}
        
        def backtrack(cell):
            if cell in path_cells:
                return
            path_cells.add(cell)
            path_frequency[cell] = path_frequency.get(cell, 0) + 1
            
            if cell in parents:
                for parent in parents[cell]:
                    backtrack(parent)
        
        for goal_cell in goal_cells:
            backtrack(goal_cell)
        
        piece_type = "B" if opponent_type == "R" else "R"
        cut_points = []
        
        for cell in path_cells:
            if cell not in env:
                i, j = cell
                priority = 0.0
                
                frequency = path_frequency.get(cell, 1)
                priority += frequency * 2.0
                
                if opponent_type == "R":
                    opp_progress = max((pos[0] for pos, p in env.items() 
                                       if p.get_type() == opponent_type), default=0)
                    dist_from_front = abs(i - opp_progress)
                    priority += max(0, 5.0 - dist_from_front * 0.5)
                else:
                    opp_progress = max((pos[1] for pos, p in env.items() 
                                       if p.get_type() == opponent_type), default=0)
                    dist_from_front = abs(j - opp_progress)
                    priority += max(0, 5.0 - dist_from_front * 0.5)
                
                if cell in distances:
                    remaining_dist = distances[cell]
                    urgency = (dim - remaining_dist) / dim
                    priority += urgency * 3.0
                
                _, corridor_strength = self.is_in_opponent_corridor(state, cell, opponent_type)
                if corridor_strength > 0.5:
                    priority *= (1.0 + corridor_strength * 0.5)
                
                ladder_break_value = self.detect_ladder_break_for_block(
                    state, cell, piece_type, opponent_type
                )
                priority += ladder_break_value * self.player.W_LADDER_BREAK / 10.0
                
                connectivity = self.local_connectivity_score(state, cell, piece_type)
                priority += connectivity * self.player.W_CONNECTIVITY / 10.0
                
                if frequency >= 2:
                    vital_score = self.detect_vital_point(state, cell, piece_type, opponent_type)
                    priority += vital_score * self.player.W_VITAL_POINT / 10.0
                
                cut_points.append((cell, priority))
        
        cut_points.sort(key=lambda x: -x[1])
        
        return cut_points
    
    def is_in_opponent_corridor(self, state: GameState, pos: Tuple[int, int], 
                                opponent_type: str) -> Tuple[bool, float]:
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        
        if pos in env:
            return False, 0.0
        
        distances = {}
        parents = {}
        pq = []
        
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
        else:
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
        
        min_goal_dist = float('inf')
        goal_cells = set()
        
        while pq:
            dist, i, j = heapq.heappop(pq)
            
            if (i, j) in distances and dist > distances[(i, j)]:
                continue
            
            is_goal = (opponent_type == "R" and i == dim - 1) or \
                     (opponent_type == "B" and j == dim - 1)
            
            if is_goal:
                if dist < min_goal_dist:
                    min_goal_dist = dist
                    goal_cells = {(i, j)}
                elif dist == min_goal_dist:
                    goal_cells.add((i, j))
                continue
            
            neighbors = state.get_neighbours(i, j)
            for neighbor_type, (ni, nj) in neighbors.values():
                if neighbor_type == "OUTSIDE":
                    continue
                
                if neighbor_type == opponent_type:
                    new_dist = dist
                elif neighbor_type == "EMPTY":
                    new_dist = dist + 1
                else:
                    continue
                
                if (ni, nj) not in distances or new_dist < distances[(ni, nj)]:
                    distances[(ni, nj)] = new_dist
                    parents[(ni, nj)] = [(i, j)]
                    heapq.heappush(pq, (new_dist, ni, nj))
                elif new_dist == distances[(ni, nj)]:
                    if (i, j) not in parents[(ni, nj)]:
                        parents[(ni, nj)].append((i, j))
        
        if pos not in distances:
            return False, 0.0
        
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
            min_dist_to_path = float('inf')
            for path_cell in path_cells:
                dist = abs(pos[0] - path_cell[0]) + abs(pos[1] - path_cell[1])
                min_dist_to_path = min(min_dist_to_path, dist)
            
            if min_dist_to_path <= 1:
                return True, 0.3
            else:
                return False, 0.0
        
        pos_dist = distances[pos]
        remaining_dist = min_goal_dist - pos_dist
        
        goal_proximity = 1.0 - (remaining_dist / max(1, min_goal_dist))
        
        path_count = len(parents.get(pos, [])) + 1
        chokepoint_factor = min(1.0, path_count / 3.0)
        
        corridor_strength = (goal_proximity * 0.6 + chokepoint_factor * 0.4)
        
        return True, corridor_strength
    
    def find_bridge_opportunities(self, state: GameState, piece_type: str) -> List[Tuple[Tuple[int, int], float]]:
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0
        
        opportunities = []
        
        for i in range(dim):
            for j in range(dim):
                if (i, j) in env:
                    continue
                
                bridge_value = 0.0
                
                for partner_off, carrier1_off, carrier2_off in self.player._bridge_offsets:
                    for partner_off2, c1_off2, c2_off2 in self.player._bridge_offsets:
                        partner = (i + partner_off2[0], j + partner_off2[1])
                        carrier1 = (i + c1_off2[0], j + c1_off2[1])
                        carrier2 = (i + c2_off2[0], j + c2_off2[1])
                        
                        if not self._in_bounds(partner, dim):
                            continue
                        
                        if partner in env and env[partner].get_type() == piece_type:
                            if self._in_bounds(carrier1, dim) and self._in_bounds(carrier2, dim):
                                c1_empty = carrier1 not in env
                                c2_empty = carrier2 not in env
                                
                                if c1_empty and c2_empty:
                                    value = 2.0
                                elif c1_empty or c2_empty:
                                    value = 1.0
                                else:
                                    continue
                                
                                row_span = abs(i - partner[0])
                                col_span = abs(j - partner[1])
                                
                                if piece_type == "R":
                                    if row_span > col_span:
                                        value *= 1.5
                                    elif row_span < col_span:
                                        value *= 0.7
                                else:
                                    if col_span > row_span:
                                        value *= 1.5
                                    elif col_span < row_span:
                                        value *= 0.7
                                
                                if piece_type == "R":
                                    progress = (i + partner[0]) / (2.0 * (dim - 1))
                                else:
                                    progress = (j + partner[1]) / (2.0 * (dim - 1))
                                value += progress * 0.5
                                
                                center_dist = abs(i - center) + abs(j - center)
                                value += 0.5 / (1.0 + center_dist * 0.1)
                                
                                bridge_value += value
                
                if bridge_value > 0:
                    opportunities.append(((i, j), bridge_value))
        
        opportunities.sort(key=lambda x: -x[1])
        return opportunities
    
    def fast_heuristic(self, state: GameState) -> float:
        if state.is_done():
            if state.get_player_score(self.player) > 0:
                return 1000.0
            else:
                return -1000.0
        
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0
        
        center_weight = self.get_dynamic_center_weight(state)
        
        my_score = 0.0
        opp_score = 0.0
        
        my_stones = []
        opp_stones = []
        
        for (i, j), piece in env.items():
            dist_from_center = abs(i - center) + abs(j - center)
            center_value = center_weight / (1.0 + dist_from_center * 0.1)
            
            if piece.get_type() == self.player.piece_type:
                if self.player.piece_type == "R":
                    progress = i / (dim - 1)
                else:
                    progress = j / (dim - 1)
                
                my_score += center_value + progress * self.player.W_PROGRESS
                my_stones.append((i, j))
            
            else:
                if self.player.opp_type == "R":
                    progress = i / (dim - 1)
                else:
                    progress = j / (dim - 1)
                
                opp_score += center_value + progress * self.player.W_PROGRESS
                opp_stones.append((i, j))
        
        my_bridge_bonus = self._quick_bridge_count(my_stones, env, self.player.piece_type, dim)
        opp_bridge_bonus = self._quick_bridge_count(opp_stones, env, self.player.opp_type, dim)
        
        my_score += my_bridge_bonus * 1.5
        opp_score += opp_bridge_bonus * 1.5
        
        return my_score - opp_score
    
    def _quick_bridge_count(self, stones: List[Tuple[int, int]], env: Dict,
                            piece_type: str, dim: int) -> float:
        count = 0.0
        stone_set = set(stones)
        
        for (i, j) in stones:
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
        
        return count / 2.0
    
    def accurate_heuristic(self, state: GameState) -> float:
        if state.is_done():
            if state.get_player_score(self.player) > 0:
                return 1000.0
            else:
                return -1000.0
        
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0
        
        my_dist = self.shortest_path_distance(state, self.player.piece_type, consider_bridges=True)
        opp_dist = self.shortest_path_distance(state, self.player.opp_type, consider_bridges=True)
        path_advantage = opp_dist - my_dist
        
        my_bridges = self.score_bridges(state, self.player.piece_type)
        opp_bridges = self.score_bridges(state, self.player.opp_type)
        bridge_advantage = my_bridges - opp_bridges
        
        threatened = self.find_threatened_bridges(state)
        threat_penalty = len(threatened) * 15.0
        
        effective_blocks = self.find_effective_blocking_moves(state, self.player.opp_type)
        
        closure_urgency = sum(priority for _, priority, mtype in effective_blocks[:3] if mtype == 'CLOSURE')
        preblock_urgency = sum(priority for _, priority, mtype in effective_blocks[:5] if mtype == 'PRE_BLOCK')
        block_urgency = closure_urgency * 1.5 + preblock_urgency * 0.5
        
        path_deficit = my_dist - opp_dist
        if path_deficit > 0:
            block_multiplier = 1.0 + (path_deficit * 0.15)
        else:
            block_multiplier = 1.0
        
        corridor_disruption = 0.0
        for (i, j), piece in env.items():
            if piece.get_type() == self.player.piece_type:
                in_corridor, strength = self.is_in_opponent_corridor(state, (i, j), self.player.opp_type)
                pass
        
        my_connectivity = 0.0
        stone_count = 0
        for (i, j), piece in env.items():
            if piece.get_type() == self.player.piece_type:
                neighbors = state.get_neighbours(i, j)
                friendly_count = sum(1 for ntype, _ in neighbors.values() if ntype == self.player.piece_type)
                my_connectivity += friendly_count
                stone_count += 1
        
        if stone_count > 0:
            avg_connectivity = my_connectivity / stone_count
            connectivity_advantage = avg_connectivity * 2.0
        else:
            connectivity_advantage = 0.0
        
        center_weight = self.get_dynamic_center_weight(state)
        
        my_center = 0.0
        opp_center = 0.0
        for (i, j), piece in env.items():
            dist = abs(i - center) + abs(j - center)
            value = 1.0 / (1.0 + dist * 0.1)
            if piece.get_type() == self.player.piece_type:
                my_center += value
            else:
                opp_center += value
        center_advantage = my_center - opp_center
        
        score = (self.player.W_PATH * path_advantage +
                 self.player.W_BRIDGE * bridge_advantage -
                 threat_penalty -
                 self.player.W_BLOCK * block_urgency * 0.5 * block_multiplier +
                 self.player.W_CONNECTIVITY * connectivity_advantage +
                 center_weight * center_advantage)
        
        return score
    
    def shortest_path_distance(self, state: GameState, piece_type: str,
                                 consider_bridges: bool = True) -> float:
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        visited = set()
        pq = []
        
        opp_type = "B" if piece_type == "R" else "R"
        
        bridge_connections = set()
        if consider_bridges:
            bridges = self.find_bridges(state, piece_type)
            for bridge in bridges:
                if bridge['complete'] or bridge['secured']:
                    bridge_connections.add((bridge['stone1'], bridge['stone2']))
                    bridge_connections.add((bridge['stone2'], bridge['stone1']))
        
        if piece_type == "R":
            for j in range(dim):
                if (0, j) in env and env[(0, j)].get_type() == piece_type:
                    cost = 0
                elif (0, j) not in env:
                    cost = 1
                else:
                    continue
                heapq.heappush(pq, (cost, 0, j))
        else:
            for i in range(dim):
                if (i, 0) in env and env[(i, 0)].get_type() == piece_type:
                    cost = 0
                elif (i, 0) not in env:
                    cost = 1
                else:
                    continue
                heapq.heappush(pq, (cost, i, 0))
        
        while pq:
            dist, i, j = heapq.heappop(pq)
            
            if (i, j) in visited:
                continue
            visited.add((i, j))
            
            if piece_type == "R" and i == dim - 1:
                return dist
            if piece_type == "B" and j == dim - 1:
                return dist
            
            neighbors = state.get_neighbours(i, j)
            for neighbor_type, (ni, nj) in neighbors.values():
                if (ni, nj) in visited:
                    continue
                
                if neighbor_type == "OUTSIDE":
                    continue
                elif neighbor_type == piece_type:
                    new_dist = dist
                elif neighbor_type == "EMPTY":
                    new_dist = dist + 1
                else:
                    continue
                
                heapq.heappush(pq, (new_dist, ni, nj))
            
            if consider_bridges:
                for (stone1, stone2) in bridge_connections:
                    if stone1 == (i, j) and stone2 not in visited:
                        heapq.heappush(pq, (dist, stone2[0], stone2[1]))
        
        return dim
    
    def virtual_connection_distance(self, state: GameState, piece_type: str) -> float:
        return self.shortest_path_distance(state, piece_type, consider_bridges=True)
    
    def biased_rollout(self, state: GameState) -> float:
        s = state
        steps = 0
        max_depth = 80
        
        while not s.is_done() and steps < max_depth:
            actions = list(s.get_possible_heavy_actions())
            if not actions:
                break
            
            epsilon = 0.30 if steps < 5 else (0.20 if steps < 15 else 0.10)
            
            roll = random.random()
            
            if roll < epsilon:
                action = random.choice(actions)
            
            elif roll < epsilon + 0.1 and steps < 20:
                action = self._quick_bridge_move(s, actions)
            
            else:
                action = max(actions,
                             key=lambda a: self.fast_heuristic(a.get_next_game_state()))
            
            s = action.get_next_game_state()
            steps += 1
        
        return self.fast_heuristic(s)
    
    def _quick_bridge_move(self, state: GameState, actions: List) -> 'Action':
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        current_player = state.get_next_player()
        piece_type = current_player.get_piece_type()
        
        if piece_type == self.player.piece_type:
            urgent = self.find_urgent_bridge_responses(state)
            if urgent:
                urgent_pos = urgent[0][0]
                for action in actions:
                    next_state = action.get_next_game_state()
                    move_pos = self._get_move_position(state, next_state)
                    if move_pos == urgent_pos:
                        return action
            
            avoid_positions = self.find_own_complete_bridge_carriers(state)
        else:
            avoid_positions = set()
        
        best_action = None
        best_bridge_score = -1
        
        sample_size = min(15, len(actions))
        sampled = random.sample(actions, sample_size)
        
        for action in sampled:
            next_state = action.get_next_game_state()
            move_pos = self._get_move_position(state, next_state)
            
            if move_pos in avoid_positions:
                continue
            
            score = 0
            for di, dj in [(1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1)]:
                partner = (move_pos[0] + di, move_pos[1] + dj)
                if partner in env and env[partner].get_type() == piece_type:
                    score += 1
            
            if score > best_bridge_score:
                best_bridge_score = score
                best_action = action
        
        return best_action if best_action else random.choice(actions)
    
    def ordered_root_actions(self, state: GameState) -> List:
        actions = list(state.get_possible_heavy_actions())
        if not actions:
            return actions
        
        env = state.get_rep().get_env()
        
        my_dist = self.shortest_path_distance(state, self.player.piece_type, consider_bridges=True)
        opp_dist = self.shortest_path_distance(state, self.player.opp_type, consider_bridges=True)
        path_deficit = my_dist - opp_dist
        
        if path_deficit > 2:
            block_bonus = self.player.BONUS_CUT * (1.0 + path_deficit * 0.2)
            bridge_bonus = self.player.BONUS_BRIDGE * 0.5
        elif path_deficit > 0:
            block_bonus = self.player.BONUS_CUT * 1.2
            bridge_bonus = self.player.BONUS_BRIDGE * 0.8
        else:
            block_bonus = self.player.BONUS_CUT
            bridge_bonus = self.player.BONUS_BRIDGE
        
        urgent_responses = {pos for pos, _ in self.find_urgent_bridge_responses(state)}
        own_complete_carriers = self.find_own_complete_bridge_carriers(state)
        
        opp_bridge_carriers = self.get_opponent_bridge_carriers(state, self.player.opp_type)
        
        effective_blocks = self.find_effective_blocking_moves(state, self.player.opp_type)
        
        block_priorities = {}
        block_metadata = {}
        closure_moves = set()
        preblock_moves = set()
        
        for pos, priority, move_type in effective_blocks:
            block_priorities[pos] = priority
            block_metadata[pos] = {'type': move_type, 'priority': priority}
            if move_type == 'CLOSURE':
                closure_moves.add(pos)
            else:
                preblock_moves.add(pos)
        
        bridge_opps = {pos for pos, _ in self.find_bridge_opportunities(state, self.player.piece_type)[:10]}
        bridge_opps = bridge_opps - own_complete_carriers
        bridge_opps = bridge_opps - opp_bridge_carriers
        
        urgent_defense = []
        wins = []
        block_closures = []
        pre_blocks = []
        bridge_moves = []
        others = []
        wasteful = []
        
        for action in actions:
            next_state = action.get_next_game_state()
            move_pos = self._get_move_position(state, next_state)
            
            if move_pos in urgent_responses:
                value = self.player.W_BRIDGE_DEFENSE
                urgent_defense.append((action, value))
            
            elif next_state.is_done() and next_state.get_player_score(self.player) > 0:
                wins.append((action, float('inf')))
            
            elif move_pos in own_complete_carriers:
                value = self.accurate_heuristic(next_state) - 50.0
                wasteful.append((action, value))
            
            elif move_pos in opp_bridge_carriers:
                value = self.accurate_heuristic(next_state) - 30.0
                wasteful.append((action, value))
            
            elif move_pos in closure_moves:
                value = self.accurate_heuristic(next_state)
                priority = block_priorities.get(move_pos, 1.0)
                value += block_bonus * (priority / 10.0) * 1.5
                block_closures.append((action, value))
            
            elif move_pos in preblock_moves:
                value = self.accurate_heuristic(next_state)
                priority = block_priorities.get(move_pos, 1.0)
                value += block_bonus * (priority / 10.0)
                pre_blocks.append((action, value))
            
            elif move_pos in bridge_opps:
                value = self.accurate_heuristic(next_state)
                value += bridge_bonus
                bridge_moves.append((action, value))
            
            else:
                value = self.accurate_heuristic(next_state)
                others.append((action, value))
        
        urgent_defense.sort(key=lambda x: -x[1])
        block_closures.sort(key=lambda x: -x[1])
        pre_blocks.sort(key=lambda x: -x[1])
        bridge_moves.sort(key=lambda x: -x[1])
        others.sort(key=lambda x: -x[1])
        wasteful.sort(key=lambda x: -x[1])
        
        all_moves = urgent_defense + wins + block_closures + pre_blocks + bridge_moves + others + wasteful
        
        if len(all_moves) > 0:
            top_action, top_value = all_moves[0]
            top_pos = self._get_move_position(state, top_action.get_next_game_state())
            category = "URGENT_DEFENSE" if top_action in [a for a, _ in urgent_defense] else \
                      "WIN" if top_action in [a for a, _ in wins] else \
                      "BLOCK_CLOSURE" if top_action in [a for a, _ in block_closures] else \
                      "PRE_BLOCK" if top_action in [a for a, _ in pre_blocks] else \
                      "BRIDGE" if top_action in [a for a, _ in bridge_moves] else \
                      "OTHER" if top_action in [a for a, _ in others] else "WASTEFUL"
            
            metadata_str = ""
            if top_pos in block_metadata:
                meta = block_metadata[top_pos]
                metadata_str = f" [{meta['type']}: {meta['priority']:.1f}]"
            
            print(f"🎲 Top move: {top_pos} [{category}] (value: {top_value:.2f}){metadata_str}")
            
            print(f"📊 Move distribution: {len(urgent_defense)} urgent, {len(wins)} wins, "
                  f"{len(block_closures)} closures, {len(pre_blocks)} pre-blocks, {len(bridge_moves)} bridges, "
                  f"{len(others)} others, {len(wasteful)} wasteful")
        
        return [action for action, _ in all_moves]
    
    def _get_move_position(self, state: GameState, next_state: GameState) -> Tuple[int, int]:
        current_env = state.get_rep().get_env()
        next_env = next_state.get_rep().get_env()
        
        for pos in next_env:
            if pos not in current_env:
                return pos
        
        return (-1, -1)
    
    def mcts_select(self, node: MCTSNode) -> MCTSNode:
        current = node
        
        while not current.is_terminal:
            if not current.is_fully_expanded():
                return current
            else:
                current = current.best_child(c=1.4)
        
        return current
    
    def mcts_expand(self, node: MCTSNode) -> MCTSNode:
        if node.is_terminal:
            return node
        
        unexpanded = node.get_unexpanded_actions()
        if not unexpanded:
            return node
        
        if len(unexpanded) > 10 and node.player_ref:
            action_scores = []
            for action in unexpanded:
                next_state = action.get_next_game_state()
                score = node.player_ref._helper.accurate_heuristic(next_state)
                action_scores.append((action, score))
            
            action_scores.sort(key=lambda x: -x[1])
            action = action_scores[0][0]
        else:
            action = unexpanded[0]
        
        node.get_unexpanded_actions().remove(action)
        
        next_state = action.get_next_game_state()
        child = MCTSNode(next_state, parent=node, action=action, player_ref=node.player_ref)
        node.children[action] = child
        
        return child
    
    def mcts_simulate(self, node: MCTSNode) -> float:
        if node.player_ref is None:
            return self.biased_rollout(node.state)
        
        if node.is_terminal:
            if node.state.get_player_score(node.player_ref) > 0:
                return 1000.0
            else:
                return -1000.0
        
        score = node.player_ref._helper.accurate_heuristic(node.state)
        
        return score
    
    def mcts_backpropagate(self, node: MCTSNode, value: float):
        current = node
        
        while current is not None:
            current.update(value)
            current = current.parent
            
            value = -value
    
    def detect_ladder(self, state: GameState, piece_type: str,
                      start_pos: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        opp_type = "B" if piece_type == "R" else "R"
        
        visited = set()
        ladder_path = []
        
        def can_force(pos: Tuple[int, int], depth: int) -> bool:
            if depth > 15:
                return False
            
            i, j = pos
            visited.add(pos)
            ladder_path.append(pos)
            
            if piece_type == "R" and i == dim - 1:
                return True
            if piece_type == "B" and j == dim - 1:
                return True
            
            neighbors = state.get_neighbours(i, j)
            forcing_options = []
            
            for direction, (ntype, (ni, nj)) in neighbors.items():
                if (ni, nj) in visited:
                    continue
                
                if ntype == "OUTSIDE":
                    continue
                elif ntype == piece_type:
                    forcing_options.append(((ni, nj), 0))
                elif ntype == "EMPTY":
                    if self._is_forcing_move((ni, nj), piece_type, env, dim):
                        forcing_options.append(((ni, nj), 1))
            
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
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        if pos in env:
            return 0.0
        
        neighbors = state.get_neighbours(pos[0], pos[1])
        opp_neighbors = []
        
        for direction, (neighbor_type, (ni, nj)) in neighbors.items():
            if neighbor_type == opponent_type:
                opp_neighbors.append((ni, nj))
        
        if len(opp_neighbors) < 2:
            return 0.0
        
        ladder_value = 0.0
        
        if opponent_type == "R":
            opp_rows = [ni for ni, nj in opp_neighbors]
            avg_row = sum(opp_rows) / len(opp_rows)
            progress = avg_row / (dim - 1)
            
            if progress > 0.5:
                ladder_value += 3.0 * progress
                
            if max(opp_rows) - min(opp_rows) >= 2:
                ladder_value += 2.0
        else:
            opp_cols = [nj for ni, nj in opp_neighbors]
            avg_col = sum(opp_cols) / len(opp_cols)
            progress = avg_col / (dim - 1)
            
            if progress > 0.5:
                ladder_value += 3.0 * progress
                
            if max(opp_cols) - min(opp_cols) >= 2:
                ladder_value += 2.0
        
        opp_set = set(opp_neighbors)
        for opp_pos in opp_neighbors:
            opp_neighs = state.get_neighbours(opp_pos[0], opp_pos[1])
            connected_count = 0
            for _, (n_type, n_pos) in opp_neighs.items():
                if n_type == opponent_type and n_pos in opp_set and n_pos != opp_pos:
                    connected_count += 1
            
            if connected_count == 0:
                ladder_value += 2.0
        
        return ladder_value
    
    def get_opponent_bridge_carriers(self, state: GameState, opponent_type: str) -> Set[Tuple[int, int]]:
        """Get all carrier cells of opponent's complete bridges - these are BAD blocking positions."""
        bridges = self.find_bridges(state, opponent_type)
        carriers = set()
        
        for bridge in bridges:
            if bridge['complete']:
                carriers.add(bridge['carrier1'])
                carriers.add(bridge['carrier2'])
        
        return carriers
    
    def get_opponent_front_line(self, state: GameState, opponent_type: str) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Find the opponent's most advanced position (front line).
        Returns: (front_line_progress, list of cells at or near the front)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        
        opp_stones = [(pos, piece) for pos, piece in env.items() if piece.get_type() == opponent_type]
        
        if not opp_stones:
            return (0, [])
        
        if opponent_type == "R":
            max_progress = max(pos[0] for pos, _ in opp_stones)
            front_stones = [pos for pos, _ in opp_stones if pos[0] >= max_progress - 1]
        else:
            max_progress = max(pos[1] for pos, _ in opp_stones)
            front_stones = [pos for pos, _ in opp_stones if pos[1] >= max_progress - 1]
        
        return (max_progress, front_stones)
    
    def find_forward_block_positions(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]:
        """
        TURN 1 - Forward "Pre-Block" Placement
        
        Find positions 2-3 cells AHEAD of opponent's front line for pre-emptive blocking.
        These positions should:
        - Be 2-3 cells ahead of opponent's most advanced stone
        - Prefer edge positions (where connection freedom is reduced)
        - NOT be adjacent to any current opponent stones
        - NOT be in bridge carrier cells (useless blocks)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        piece_type = "B" if opponent_type == "R" else "R"
        
        front_progress, front_stones = self.get_opponent_front_line(state, opponent_type)
        
        if not front_stones:
            return []
        
        bridge_carriers = self.get_opponent_bridge_carriers(state, opponent_type)
        
        candidates = []
        
        for i in range(dim):
            for j in range(dim):
                if (i, j) in env:
                    continue
                
                if (i, j) in bridge_carriers:
                    continue
                
                if opponent_type == "R":
                    distance_ahead = i - front_progress
                    is_edge = (j == 0 or j == dim - 1)
                else:
                    distance_ahead = j - front_progress
                    is_edge = (i == 0 or i == dim - 1)
                
                if distance_ahead < 2 or distance_ahead > 4:
                    continue
                
                neighbors = state.get_neighbours(i, j)
                adjacent_to_opponent = False
                for _, (n_type, _) in neighbors.items():
                    if n_type == opponent_type:
                        adjacent_to_opponent = True
                        break
                
                if adjacent_to_opponent:
                    continue
                
                priority = 10.0
                
                if distance_ahead == 2:
                    priority += 4.0
                elif distance_ahead == 3:
                    priority += 3.0
                else:
                    priority += 1.0
                
                if is_edge:
                    priority += 5.0
                
                if opponent_type == "R":
                    front_cols = [pos[1] for pos in front_stones]
                    avg_col = sum(front_cols) / len(front_cols)
                    col_alignment = 1.0 / (1.0 + abs(j - avg_col))
                    priority += col_alignment * 3.0
                else:
                    front_rows = [pos[0] for pos in front_stones]
                    avg_row = sum(front_rows) / len(front_rows)
                    row_alignment = 1.0 / (1.0 + abs(i - avg_row))
                    priority += row_alignment * 3.0
                
                connectivity = self.local_connectivity_score(state, (i, j), piece_type)
                priority += connectivity * 0.5
                
                candidates.append(((i, j), priority))
        
        candidates.sort(key=lambda x: -x[1])
        return candidates
    
    def find_block_closure_moves(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float]]:
        """
        TURN 2 - Block Closure
        
        Find moves that complete a blocking wall by connecting our pre-block stones.
        Looks for our stones that are:
        - Ahead of the opponent's front line
        - Can be connected to create a wall (U-Block or Classic Block formation)
        """
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        piece_type = "B" if opponent_type == "R" else "R"
        
        front_progress, front_stones = self.get_opponent_front_line(state, opponent_type)
        
        if not front_stones:
            return []
        
        our_blocking_stones = []
        for pos, piece in env.items():
            if piece.get_type() != piece_type:
                continue
            
            if opponent_type == "R":
                if pos[0] >= front_progress:
                    our_blocking_stones.append(pos)
            else:
                if pos[1] >= front_progress:
                    our_blocking_stones.append(pos)
        
        if not our_blocking_stones:
            return []
        
        bridge_carriers = self.get_opponent_bridge_carriers(state, opponent_type)
        
        closure_candidates = []
        
        for stone in our_blocking_stones:
            neighbors = state.get_neighbours(stone[0], stone[1])
            
            for _, (n_type, (ni, nj)) in neighbors.items():
                if n_type != "EMPTY":
                    continue
                
                if (ni, nj) in bridge_carriers:
                    continue
                
                priority = 15.0
                
                for other_stone in our_blocking_stones:
                    if other_stone == stone:
                        continue
                    
                    dist = abs(ni - other_stone[0]) + abs(nj - other_stone[1])
                    if dist <= 2:
                        priority += 10.0 / dist
                
                if opponent_type == "R":
                    is_edge = (nj == 0 or nj == dim - 1)
                    dist_to_edge = min(nj, dim - 1 - nj)
                else:
                    is_edge = (ni == 0 or ni == dim - 1)
                    dist_to_edge = min(ni, dim - 1 - ni)
                
                if is_edge:
                    priority += 8.0
                elif dist_to_edge <= 2:
                    priority += 4.0
                
                for other_stone in our_blocking_stones:
                    for partner_off, c1_off, c2_off in self.player._bridge_offsets:
                        partner = (ni + partner_off[0], nj + partner_off[1])
                        carrier1 = (ni + c1_off[0], nj + c1_off[1])
                        carrier2 = (ni + c2_off[0], nj + c2_off[1])
                        
                        if partner == other_stone:
                            if self._in_bounds(carrier1, dim) and self._in_bounds(carrier2, dim):
                                c1_empty = carrier1 not in env
                                c2_empty = carrier2 not in env
                                if c1_empty and c2_empty:
                                    priority += 6.0
                
                existing = next((p for p, _ in closure_candidates if p == (ni, nj)), None)
                if existing is None:
                    closure_candidates.append(((ni, nj), priority))
                else:
                    idx = next(i for i, (p, _) in enumerate(closure_candidates) if p == (ni, nj))
                    if priority > closure_candidates[idx][1]:
                        closure_candidates[idx] = ((ni, nj), priority)
        
        closure_candidates.sort(key=lambda x: -x[1])
        return closure_candidates
    
    def find_effective_blocking_moves(self, state: GameState, opponent_type: str) -> List[Tuple[Tuple[int, int], float, str]]:
        """
        Main blocking strategy that combines:
        1. Block closure moves (if we have pre-blocks in place) - highest priority
        2. Forward pre-block positions (for setting up future blocks)
        
        Returns: List of (position, priority, move_type) where move_type is 'CLOSURE' or 'PRE_BLOCK'
        """
        closure_moves = self.find_block_closure_moves(state, opponent_type)
        forward_moves = self.find_forward_block_positions(state, opponent_type)
        
        result = []
        
        for pos, priority in closure_moves[:5]:
            result.append((pos, priority + 20.0, 'CLOSURE'))
        
        for pos, priority in forward_moves[:10]:
            already_added = any(p == pos for p, _, _ in result)
            if not already_added:
                result.append((pos, priority, 'PRE_BLOCK'))
        
        result.sort(key=lambda x: -x[1])
        return result
    
    def _is_forcing_move(self, pos: Tuple[int, int], piece_type: str,
                         env: Dict, dim: int) -> bool:
        i, j = pos
        
        threat_count = 0
        
        offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < dim and 0 <= nj < dim:
                if (ni, nj) in env and env[(ni, nj)].get_type() == piece_type:
                    threat_count += 1
        
        if piece_type == "R":
            goal_dist = dim - 1 - i
        else:
            goal_dist = dim - 1 - j
        
        return threat_count >= 1 and goal_dist <= 4
    
    def _get_tree_depth(self, node: MCTSNode, current_depth: int = 0) -> int:
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_tree_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth

