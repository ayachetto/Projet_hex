# Comprehensive Hex Game Implementation Documentation

## Table of Contents
1. [Game Overview](#game-overview)
2. [What the Computer Sees](#what-the-computer-sees)
3. [Game State Representation](#game-state-representation)
4. [Actions and Moves](#actions-and-moves)
5. [Win Conditions](#win-conditions)
6. [Board Topology and Navigation](#board-topology-and-navigation)
7. [Creating Effective Heuristics](#creating-effective-heuristics)
8. [Strategic Concepts for Hex](#strategic-concepts-for-hex)
9. [Why Your Current Implementation May Be Underperforming](#why-your-current-implementation-may-be-underperforming)
10. [Recommended Heuristics](#recommended-heuristics)

---

## 1. Game Overview

### Hex Game Rules
- **Board**: An N×N hexagonal grid (typically 11×11 or 14×14)
- **Players**: 
  - Player 1 (Red/"R"): Connects TOP to BOTTOM
  - Player 2 (Blue/"B"): Connects LEFT to RIGHT
- **Winning**: First player to create an unbroken path of their pieces connecting their two sides wins
- **No Draws**: Hex cannot end in a draw - someone always wins

### Player Objectives
- **Red (R)**: Must connect row 0 (top) to row N-1 (bottom)
- **Blue (B)**: Must connect column 0 (left) to column N-1 (right)

---

## 2. What the Computer Sees

### A. Game State Information (`current_state: GameStateHex`)

When your AI's `compute_action()` is called, you receive a `GameStateHex` object containing:

```python
# Core information available
current_state.get_rep()                    # BoardHex object
current_state.get_players()                # List of players [player1, player2]
current_state.get_next_player()            # Whose turn it is
current_state.get_scores()                 # Dict {player_id: score}
current_state.is_done()                    # Boolean: is game over?
current_state.get_step()                   # Current move number
current_state.max_step                     # Max possible moves (N*N)
```

### B. Board Representation (`current_state.get_rep()`)

The board is represented as a dictionary mapping positions to pieces:

```python
board = current_state.get_rep()
env = board.get_env()                      # Dict: {(i,j): Piece}
dimensions = board.get_dimensions()        # Tuple: (N, N)

# Example structure:
# env = {
#     (0, 0): Piece(piece_type="R", owner=player1),
#     (0, 1): Piece(piece_type="B", owner=player2),
#     (1, 1): Piece(piece_type="R", owner=player1),
#     # ... only occupied cells are in the dictionary
# }
```

**Key Points:**
- Only **occupied** cells exist in the dictionary
- Empty cells are **NOT** in the dictionary
- Check if cell is empty: `(i, j) not in env`
- Check if cell is occupied: `(i, j) in env`
- Get piece type: `env[(i, j)].get_type()` → returns "R" or "B"

### C. Iterating Over the Board

```python
# Get board dimensions
dim = current_state.get_rep().get_dimensions()[0]  # e.g., 11

# Iterate over ALL cells (occupied and empty)
for i in range(dim):
    for j in range(dim):
        if (i, j) in env:
            piece = env[(i, j)]
            piece_type = piece.get_type()  # "R" or "B"
            owner = piece.owner            # Player object
        else:
            # Cell is empty
            pass

# Or get only empty cells
empty_cells = list(board.get_empty())  # Generator of (i,j) tuples
```

### D. Getting Available Actions

```python
# Light actions (just the move data)
light_actions = list(current_state.get_possible_light_actions())
# Each action: LightAction({"piece": "R", "position": (i, j)})

# Heavy actions (includes resulting game state)
heavy_actions = list(current_state.get_possible_heavy_actions())
# Each action: HeavyAction with .get_next_game_state()
```

---

## 3. Game State Representation

### Key Classes

#### `GameStateHex`
The complete game state containing:
- `rep`: The board (BoardHex)
- `players`: List of Player objects
- `next_player`: Who moves next
- `scores`: Dict mapping player_id → score (0.0 or 1.0)
- `step`: Current move number

#### `BoardHex`
The board representation:
- `env`: Dictionary {(row, col): Piece}
- `dimensions`: Tuple (N, N)

#### `Piece`
A single game piece:
- `piece_type`: "R" or "B"
- `owner`: Player who placed it

### Coordinate System

```
    0   1   2   3   4   ...  (columns, j)
0   ⬢   ⬢   ⬢   ⬢   ⬢
  1   ⬢   ⬢   ⬢   ⬢   ⬢
    2   ⬢   ⬢   ⬢   ⬢   ⬢
      3   ⬢   ⬢   ⬢   ⬢   ⬢
        ...
(rows, i)

- Position (i, j) means: row i, column j
- Red connects: i=0 to i=(N-1)
- Blue connects: j=0 to j=(N-1)
```

---

## 4. Actions and Moves

### A. LightAction
Just the move information:
```python
from seahorse.game.light_action import LightAction

action = LightAction({
    "piece": "R",           # or "B"
    "position": (i, j)      # tuple of ints
})
```

### B. HeavyAction
Contains both the action AND the resulting game state:
```python
for heavy_action in current_state.get_possible_heavy_actions():
    next_state = heavy_action.get_next_game_state()
    # next_state is a complete GameStateHex after the move
```

### C. How to Apply an Action

**Method 1: Using apply_action (creates new state)**
```python
light_action = LightAction({"piece": self.piece_type, "position": (5, 5)})
new_state = current_state.apply_action(light_action)
```

**Method 2: Using heavy actions (already includes next state)**
```python
heavy_action = some_heavy_action
next_state = heavy_action.get_next_game_state()
```

---

## 5. Win Conditions

### How Wins Are Detected

The `compute_scores()` method in `GameStateHex` uses **Depth-First Search (DFS)** to check if a winning path exists:

```python
# Pseudocode of win detection:

# For Red player (piece_type = "R"):
for j in range(dim):
    if there's a Red piece at (0, j):  # top row
        if dfs_can_reach_bottom(0, j):
            Red wins!  # score = 1.0

# For Blue player (piece_type = "B"):
for i in range(dim):
    if there's a Blue piece at (i, 0):  # left column
        if dfs_can_reach_right(i, 0):
            Blue wins!  # score = 1.0
```

### Understanding Connectivity
- A path must be **contiguous** (pieces touching via neighbors)
- Pieces are connected if they share a hexagonal edge (6 possible neighbors)
- The path does **NOT** need to be shortest or direct

---

## 6. Board Topology and Navigation

### A. Hexagonal Neighbors

Each cell (i, j) has **6 neighbors**:

```python
neighbors = {
    "top_left":  (i-1, j),      # ↖
    "top_right": (i-1, j+1),    # ↗
    "left":      (i,   j-1),    # ←
    "right":     (i,   j+1),    # →
    "bot_left":  (i+1, j-1),    # ↙
    "bot_right": (i+1, j),      # ↘
}
```

**Visual representation:**
```
      (i-1,j)  (i-1,j+1)
         ↖  ↗
  (i,j-1) ← ⬢ → (i,j+1)
         ↙  ↘
     (i+1,j-1) (i+1,j)
```

### B. Using get_neighbours()

```python
neighbors_dict = current_state.get_neighbours(i, j)

# Returns: Dict[str, Tuple[str, Tuple[int, int]]]
# Example:
# {
#     "top_left": ("EMPTY", (i-1, j)),
#     "top_right": ("R", (i-1, j+1)),
#     "left": ("OUTSIDE", (i, j-1)),
#     "right": ("B", (i, j+1)),
#     "bot_left": ("EMPTY", (i+1, j-1)),
#     "bot_right": ("EMPTY", (i+1, j))
# }

# Neighbor types:
# - "EMPTY": unoccupied cell
# - "R" or "B": occupied by that piece type
# - "OUTSIDE": out of bounds
```

### C. Extracting Useful Information

```python
def get_empty_neighbors(state, i, j):
    """Get all empty neighboring positions"""
    neighbors = state.get_neighbours(i, j)
    empty = []
    for direction, (neighbor_type, (ni, nj)) in neighbors.items():
        if neighbor_type == "EMPTY":
            empty.append((ni, nj))
    return empty

def get_friendly_neighbors(state, i, j, piece_type):
    """Get all neighboring positions with our piece"""
    neighbors = state.get_neighbours(i, j)
    friendly = []
    for direction, (neighbor_type, (ni, nj)) in neighbors.items():
        if neighbor_type == piece_type:
            friendly.append((ni, nj))
    return friendly
```

---

## 7. Creating Effective Heuristics

### A. What Makes a Good Heuristic?

1. **Fast to compute** (will be called thousands of times)
2. **Correlated with winning** (higher value = better position)
3. **Differentiates positions** (not all moves score the same)
4. **Considers both offense and defense**

### B. Understanding Score vs Heuristic

**Score** (game-defined):
```python
scores = current_state.get_scores()
# Returns: {player1_id: 0.0, player2_id: 0.0}  # during game
# Returns: {player1_id: 1.0, player2_id: 0.0}  # when player1 wins
```

**Heuristic** (YOU must define):
- An **estimate** of how good a position is
- Should be non-zero even in non-terminal states
- Should reflect strategic advantage

---

## 8. Strategic Concepts for Hex

### A. Key Strategic Principles

1. **Center Control**: Center positions are more valuable
   - They connect to more potential paths
   - They're harder to block

2. **Connection**: Keep your pieces connected
   - Connected groups are stronger
   - Isolated pieces are easily blocked

3. **Blocking**: Prevent opponent's connections
   - Especially near their goal sides

4. **Progress**: Advance toward your goal
   - But don't sacrifice connectivity

### B. Measuring Position Quality

#### 1. **Shortest Path Distance**
How many moves to connect your sides?

```python
def shortest_path_to_goal(state, piece_type):
    """
    Use Dijkstra/BFS to find shortest path from start side to goal side.
    Cost: 0 for own pieces, 1 for empty cells, infinity for opponent pieces.
    Returns the distance.
    """
    import heapq
    
    dim = state.get_rep().get_dimensions()[0]
    env = state.get_rep().get_env()
    dist = {}
    pq = []
    
    if piece_type == "R":
        # Start from top row (i=0)
        for j in range(dim):
            if (0, j) in env and env[(0, j)].get_type() == "R":
                cost = 0
            elif (0, j) not in env:
                cost = 1
            else:
                continue  # opponent piece, skip
            dist[(0, j)] = cost
            heapq.heappush(pq, (cost, 0, j))
        
        # Goal: reach bottom row (i=dim-1)
        goal_row = dim - 1
    else:  # Blue
        # Start from left column (j=0)
        for i in range(dim):
            if (i, 0) in env and env[(i, 0)].get_type() == "B":
                cost = 0
            elif (i, 0) not in env:
                cost = 1
            else:
                continue
            dist[(i, 0)] = cost
            heapq.heappush(pq, (cost, i, 0))
        
        # Goal: reach right column (j=dim-1)
        goal_col = dim - 1
    
    min_dist = float('inf')
    
    while pq:
        d, i, j = heapq.heappop(pq)
        
        if (i, j) in dist and d > dist[(i, j)]:
            continue
        
        # Check if reached goal
        if piece_type == "R" and i == goal_row:
            min_dist = min(min_dist, d)
            continue
        elif piece_type == "B" and j == goal_col:
            min_dist = min(min_dist, d)
            continue
        
        # Explore neighbors
        for neighbor_type, (ni, nj) in state.get_neighbours(i, j).values():
            if neighbor_type == "OUTSIDE":
                continue
            
            if neighbor_type == piece_type:
                new_dist = d  # Free to use own pieces
            elif neighbor_type == "EMPTY":
                new_dist = d + 1  # Need to place a piece
            else:
                continue  # Opponent piece, can't use
            
            if (ni, nj) not in dist or new_dist < dist[(ni, nj)]:
                dist[(ni, nj)] = new_dist
                heapq.heappush(pq, (new_dist, ni, nj))
    
    return min_dist if min_dist != float('inf') else dim
```

**Using this in heuristic:**
```python
my_dist = shortest_path_to_goal(state, self.piece_type)
opp_dist = shortest_path_to_goal(state, opponent_piece_type)
heuristic = opp_dist - my_dist  # Higher is better for us
```

#### 2. **Connected Component Size**
How many of your pieces are in the largest connected group?

```python
def largest_component_size(state, piece_type):
    """Find the largest connected group of pieces"""
    env = state.get_rep().get_env()
    dim = state.get_rep().get_dimensions()[0]
    
    visited = set()
    max_size = 0
    
    def dfs(i, j):
        if (i, j) in visited:
            return 0
        if (i, j) not in env or env[(i, j)].get_type() != piece_type:
            return 0
        
        visited.add((i, j))
        size = 1
        
        for neighbor_type, (ni, nj) in state.get_neighbours(i, j).values():
            if neighbor_type == piece_type and (ni, nj) not in visited:
                size += dfs(ni, nj)
        
        return size
    
    for i in range(dim):
        for j in range(dim):
            if (i, j) in env and env[(i, j)].get_type() == piece_type and (i, j) not in visited:
                size = dfs(i, j)
                max_size = max(max_size, size)
    
    return max_size
```

#### 3. **Center Control Score**
Value positions by distance from center:

```python
def position_value(i, j, dim):
    """Higher value for positions near center"""
    center = (dim - 1) / 2.0
    dist_from_center = abs(i - center) + abs(j - center)
    max_dist = dim - 1
    # Normalize to [0, 1], with 1 being center
    return 1.0 - (dist_from_center / (2 * max_dist))

def center_control_score(state, piece_type):
    """Sum of position values for all our pieces"""
    env = state.get_rep().get_env()
    dim = state.get_rep().get_dimensions()[0]
    
    total = 0.0
    count = 0
    for (i, j), piece in env.items():
        if piece.get_type() == piece_type:
            total += position_value(i, j, dim)
            count += 1
    
    return total / max(1, count)  # Average position value
```

#### 4. **Bridge Patterns** (Advanced)
In Hex, a "bridge" is a pattern that guarantees connection:

```
    ⬢ - - ⬢
     \ X /
      X X
```

Two pieces with two empty spaces between them in this pattern are virtually connected.

### C. Combined Heuristic Example

```python
def evaluate_position(self, state):
    """
    Comprehensive position evaluation.
    Returns a score where higher is better for self.
    """
    # If terminal, return actual result
    if state.is_done():
        if state.get_player_score(self) > 0:
            return 1000.0  # We won
        else:
            return -1000.0  # We lost
    
    my_type = self.piece_type
    opp_type = "B" if my_type == "R" else "R"
    
    # 1. Path distance (most important)
    my_path = shortest_path_to_goal(state, my_type)
    opp_path = shortest_path_to_goal(state, opp_type)
    path_advantage = (opp_path - my_path) * 10.0
    
    # 2. Connected components
    my_component = largest_component_size(state, my_type)
    opp_component = largest_component_size(state, opp_type)
    component_advantage = (my_component - opp_component) * 2.0
    
    # 3. Center control
    my_center = center_control_score(state, my_type)
    opp_center = center_control_score(state, opp_type)
    center_advantage = (my_center - opp_center) * 5.0
    
    # 4. Piece count (minor factor)
    env = state.get_rep().get_env()
    my_pieces = sum(1 for p in env.values() if p.get_type() == my_type)
    opp_pieces = sum(1 for p in env.values() if p.get_type() == opp_type)
    piece_advantage = (my_pieces - opp_pieces) * 0.5
    
    total = path_advantage + component_advantage + center_advantage + piece_advantage
    return total
```

---

## 9. Recommended Heuristics

### Option 1: Simple & Fast (For MCTS Rollouts)

```python
def fast_heuristic(self, state):
    """
    Fast heuristic for rollout policy.
    Focus on speed over accuracy.
    """
    if state.is_done():
        return 1000.0 if state.get_player_score(self) > 0 else -1000.0
    
    env = state.get_rep().get_env()
    dim = state.get_rep().get_dimensions()[0]
    center = (dim - 1) / 2.0
    
    my_score = 0.0
    opp_score = 0.0
    
    my_type = self.piece_type
    opp_type = "B" if my_type == "R" else "R"
    
    for (i, j), piece in env.items():
        # Center bonus
        dist_from_center = abs(i - center) + abs(j - center)
        center_value = 1.0 / (1.0 + dist_from_center * 0.2)
        
        # Progress bonus
        if piece.get_type() == my_type:
            if my_type == "R":
                progress = i / (dim - 1)  # How far down
            else:
                progress = j / (dim - 1)  # How far right
            my_score += center_value + progress
        else:
            if opp_type == "R":
                progress = i / (dim - 1)
            else:
                progress = j / (dim - 1)
            opp_score += center_value + progress
    
    return my_score - opp_score
```

### Option 2: Accurate & Slower (For Move Selection)

```python
def accurate_heuristic(self, state):
    """
    More accurate heuristic for root-level move evaluation.
    Uses shortest path calculation.
    """
    if state.is_done():
        return 1000.0 if state.get_player_score(self) > 0 else -1000.0
    
    my_type = self.piece_type
    opp_type = "B" if my_type == "R" else "R"
    
    # Calculate shortest paths
    my_dist = self.shortest_path_distance(state, my_type)
    opp_dist = self.shortest_path_distance(state, opp_type)
    
    # Winning is being closer to goal
    path_diff = (opp_dist - my_dist) * 5.0
    
    return path_diff

def shortest_path_distance(self, state, piece_type):
    """Simplified Dijkstra for path distance"""
    import heapq
    
    dim = state.get_rep().get_dimensions()[0]
    env = state.get_rep().get_env()
    visited = set()
    pq = []
    
    # Initialize starting positions
    if piece_type == "R":
        for j in range(dim):
            cost = 0 if (0,j) in env and env[(0,j)].get_type() == piece_type else 1
            if (0,j) not in env or env[(0,j)].get_type() != self.get_opponent_type(piece_type):
                heapq.heappush(pq, (cost, 0, j))
    else:
        for i in range(dim):
            cost = 0 if (i,0) in env and env[(i,0)].get_type() == piece_type else 1
            if (i,0) not in env or env[(i,0)].get_type() != self.get_opponent_type(piece_type):
                heapq.heappush(pq, (cost, i, 0))
    
    while pq:
        dist, i, j = heapq.heappop(pq)
        
        if (i, j) in visited:
            continue
        visited.add((i, j))
        
        # Check if reached goal
        if piece_type == "R" and i == dim - 1:
            return dist
        if piece_type == "B" and j == dim - 1:
            return dist
        
        # Explore neighbors
        for neighbor_type, (ni, nj) in state.get_neighbours(i, j).values():
            if (ni, nj) in visited or neighbor_type == "OUTSIDE":
                continue
            
            if neighbor_type == piece_type:
                new_dist = dist
            elif neighbor_type == "EMPTY":
                new_dist = dist + 1
            else:  # Opponent piece
                continue
            
            heapq.heappush(pq, (new_dist, ni, nj))
    
    return dim  # No path found (shouldn't happen in valid Hex)

def get_opponent_type(self, piece_type):
    return "B" if piece_type == "R" else "R"
```

### Option 3: Hybrid Approach (Recommended)

```python
class MyPlayer(PlayerHex):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self.opp_type = "B" if piece_type == "R" else "R"
    
    def fast_heuristic(self, state):
        """Fast evaluation for rollouts (called thousands of times)"""
        if state.is_done():
            return 1000.0 if state.get_player_score(self) > 0 else -1000.0
        
        env = state.get_rep().get_env()
        dim = state.get_rep().get_dimensions()[0]
        center = (dim - 1) / 2.0
        
        my_score = 0.0
        opp_score = 0.0
        
        for (i, j), piece in env.items():
            dist_from_center = abs(i - center) + abs(j - center)
            center_val = 2.0 / (1.0 + dist_from_center * 0.1)
            
            if piece.get_type() == self.piece_type:
                progress = (i if self.piece_type == "R" else j) / dim
                my_score += center_val + progress * 2.0
            else:
                progress = (i if self.opp_type == "R" else j) / dim
                opp_score += center_val + progress * 2.0
        
        return my_score - opp_score
    
    def slow_heuristic(self, state):
        """Accurate evaluation for root move ordering (called ~100 times)"""
        if state.is_done():
            return 1000.0 if state.get_player_score(self) > 0 else -1000.0
        
        # Use shortest path
        my_dist = self.shortest_path_distance(state, self.piece_type)
        opp_dist = self.shortest_path_distance(state, self.opp_type)
        
        return (opp_dist - my_dist) * 10.0
    
    # ... implement shortest_path_distance as shown above ...
    
    def _random_rollout(self, state):
        """Use fast heuristic for rollouts"""
        s = state
        steps = 0
        while not s.is_done() and steps < 100:  # Add max depth
            actions = list(s.get_possible_heavy_actions())
            if not actions:
                break
            
            # Epsilon-greedy with fast heuristic
            if random.random() < 0.2:
                a = random.choice(actions)
            else:
                a = max(actions, key=lambda act: self.fast_heuristic(act.get_next_game_state()))
            
            s = a.get_next_game_state()
            steps += 1
        
        return self.fast_heuristic(s)  # Return heuristic value
    
    def _ordered_root_actions(self, state):
        """Use slow but accurate heuristic for root ordering"""
        acts = list(state.get_possible_heavy_actions())
        
        # Check for immediate wins first
        wins = []
        others = []
        
        for a in acts:
            ns = a.get_next_game_state()
            if ns.is_done() and state.get_player_score(self) > 0:
                wins.append((a, 1000.0))
            else:
                others.append((a, self.slow_heuristic(ns)))
        
        others.sort(key=lambda x: -x[1])
        return [a for a, _ in wins + others]
```

---

## Summary: Key Takeaways

### Best Strategy for Hex

1. **Opening**: Play near center
2. **Mid-game**: Maintain connected groups, block opponent
3. **End-game**: Complete shortest path to goal

### Quick Wins

```python
# Replace your _heuristic_value with:
def _heuristic_value(self, next_state):
    if next_state.is_done():
        return 1000.0 if next_state.get_player_score(self) > 0 else -1000.0
    
    # Simple but effective: count path distance
    env = next_state.get_rep().get_env()
    dim = next_state.get_rep().get_dimensions()[0]
    
    score = 0.0
    for (i, j), piece in env.items():
        if piece.get_type() == self.piece_type:
            # Progress toward goal
            progress = (i if self.piece_type == "R" else j)
            score += progress / dim
    
    return score
```

This simple change should make your player competitive with random!

---

## Additional Resources

### Debugging Your Heuristic

```python
def debug_heuristic(self, state):
    """Print heuristic components for analysis"""
    if state.is_done():
        print(f"Terminal: {state.get_player_score(self)}")
        return
    
    env = state.get_rep().get_env()
    my_pieces = sum(1 for p in env.values() if p.get_type() == self.piece_type)
    opp_pieces = len(env) - my_pieces
    
    heuristic = self._heuristic_value(state)
    
    print(f"Pieces: {my_pieces} vs {opp_pieces}")
    print(f"Heuristic: {heuristic:.2f}")
    print(f"Board:\n{state.get_rep()}")
```

### Testing Positions

```python
# Create a test position to evaluate
from board_hex import BoardHex
from seahorse.game.game_layout.board import Piece

# Create specific board position for testing
env = {
    (0, 5): Piece(piece_type="R", owner=player1),
    (5, 5): Piece(piece_type="R", owner=player1),
    (5, 0): Piece(piece_type="B", owner=player2),
}
board = BoardHex(env=env, dim=11)
# ... create game state and evaluate
```

Good luck with your Hex AI! The key is having a heuristic that actually differentiates positions, not one that returns 0 for everything.

