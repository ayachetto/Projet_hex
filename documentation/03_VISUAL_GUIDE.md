# Visual Guide to Hex Game AI

## 🎯 The Problem with Your Current Code

---

## 🎮 How the Game Actually Works

### Board Representation

```
What you see:              What the computer sees:
    
  ⬢ ⬢ ⬢ ⬢ ⬢               env = {
   R ⬢ ⬢ ⬢ ⬢                  (1, 0): Piece(type="R"),
    ⬢ ⬢ B ⬢ ⬢      →          (2, 2): Piece(type="B"),
     ⬢ ⬢ ⬢ ⬢ ⬢                (3, 3): Piece(type="R")
      ⬢ ⬢ ⬢ R ⬢            }
```

**Key Insight:** Empty cells are **NOT** in the dictionary!

```python
# Check if position (i, j) is empty:
if (i, j) not in env:
    # Empty!
    
# Check if occupied:
if (i, j) in env:
    piece_type = env[(i, j)].get_type()  # "R" or "B"
```

---

## 🗺️ Hex Neighbors (6 directions)

```
Each hex has 6 neighbors:

       (i-1, j)  (i-1, j+1)
           ↖  ↗
    (i,j-1) ← 🔴 → (i, j+1)
           ↙  ↘
      (i+1, j-1)  (i+1, j)
```

### Using get_neighbours()

```python
neighbors = state.get_neighbours(i, j)

# Returns:
{
    "top_left":  ("EMPTY", (i-1, j)),
    "top_right": ("R", (i-1, j+1)),      # Red piece
    "left":      ("OUTSIDE", (i, j-1)),  # Off board
    "right":     ("B", (i, j+1)),        # Blue piece
    "bot_left":  ("EMPTY", (i+1, j-1)),
    "bot_right": ("EMPTY", (i+1, j))
}
```

---

## 🏆 Win Conditions

### Red Wins: Connect TOP to BOTTOM

```
     0  1  2  3  4
  0  R ⬢ ⬢ ⬢ ⬢  ← Start here (row 0)
   1  ⬢ R ⬢ ⬢ ⬢
    2  ⬢ ⬢ R ⬢ ⬢
     3  ⬢ ⬢ ⬢ R ⬢
      4  ⬢ ⬢ ⬢ ⬢ R  ← Goal: reach row N-1
      
Path exists: Red wins!
```

### Blue Wins: Connect LEFT to RIGHT

```
     0  1  2  3  4
  0  B ⬢ ⬢ ⬢ ⬢
   1  B B B ⬢ ⬢
    2  ⬢ ⬢ ⬢ B ⬢
     3  ⬢ ⬢ ⬢ ⬢ B
      4  ⬢ ⬢ ⬢ ⬢ B
      ^           ^
     Start      Goal
  (column 0)  (column N-1)
  
Path exists: Blue wins!
```

---

## 💡 What Makes a Good Position?

### ❌ Bad Heuristic 

```python
score = state.get_player_score(self) - state.get_player_score(opponent)
# Always 0.0 until game ends!

Result:
Position A: score = 0.0
Position B: score = 0.0
Position C: score = 0.0
→ All positions look the same!
```

### ✅ Good Heuristic

Different positions should have different values:

```
Position A (good):          Position B (better):
  R ⬢ ⬢ ⬢ ⬢                  R ⬢ ⬢ ⬢ ⬢
   ⬢ ⬢ ⬢ ⬢ ⬢                   R ⬢ ⬢ ⬢ ⬢
    ⬢ ⬢ ⬢ ⬢ ⬢                   R ⬢ ⬢ ⬢ ⬢
     ⬢ ⬢ ⬢ ⬢ ⬢                   R ⬢ ⬢ ⬢ ⬢
      ⬢ ⬢ ⬢ ⬢ ⬢                   ⬢ ⬢ ⬢ ⬢ ⬢

Score: 0.0 (BAD)           Score: 0.8 (GOOD)
Need 3 more pieces         Need 1 more piece!
```

---

## 📊 Heuristic Components

### 1. Path Distance (Most Important!)

**Concept:** How many empty cells do you need to fill to win?

```python
# Red needs to connect top to bottom
# Count minimum empty cells in best path

Position 1:                Position 2:
  R ⬢ ⬢ ⬢ ⬢                 R ⬢ ⬢ ⬢ ⬢
   ⬢ ⬢ ⬢ ⬢ ⬢                 R ⬢ ⬢ ⬢ ⬢
    ⬢ ⬢ ⬢ ⬢ ⬢                 ⬢ ⬢ ⬢ ⬢ ⬢
     ⬢ ⬢ ⬢ ⬢ ⬢                 R ⬢ ⬢ ⬢ ⬢
      ⬢ ⬢ ⬢ ⬢ ⬢                 R ⬢ ⬢ ⬢ ⬢

Distance: 3 cells          Distance: 1 cell
Heuristic: -3.0            Heuristic: -1.0  ← BETTER!
```

**Formula:**
```python
my_distance = count_empty_cells_in_best_path(state, my_color)
opp_distance = count_empty_cells_in_best_path(state, opp_color)

heuristic = opp_distance - my_distance
# Positive = we're closer to winning
# Negative = opponent is closer
```

### 2. Center Control

**Concept:** Center positions are more valuable (more connections)

```
Center positions (weight = 1.0):
     ⬢ ⬢ ⬢ ⬢ ⬢
      ⬢ ⬢ ⬢ ⬢ ⬢
       ⬢ ⬢ 🟡 ⬢ ⬢  ← Center
        ⬢ ⬢ ⬢ ⬢ ⬢
         ⬢ ⬢ ⬢ ⬢ ⬢

Edge positions (weight = 0.3):
     🔴 ⬢ ⬢ ⬢ ⬢  ← Edge
      ⬢ ⬢ ⬢ ⬢ ⬢
       ⬢ ⬢ ⬢ ⬢ ⬢
        ⬢ ⬢ ⬢ ⬢ ⬢
         ⬢ ⬢ ⬢ ⬢ ⬢
```

**Formula:**
```python
center = (dim - 1) / 2.0  # e.g., 5.0 for 11x11 board
distance_from_center = abs(i - center) + abs(j - center)
weight = 1.0 / (1.0 + distance_from_center * 0.1)

# Sum weights for all your pieces
```

### 3. Connectivity

**Concept:** Connected pieces are stronger than isolated pieces

```
Good (connected):          Bad (isolated):
  R R ⬢ ⬢ ⬢                 R ⬢ ⬢ R ⬢
   ⬢ R ⬢ ⬢ ⬢                 ⬢ ⬢ ⬢ ⬢ ⬢
    ⬢ R ⬢ ⬢ ⬢                 R ⬢ ⬢ ⬢ ⬢
     ⬢ ⬢ ⬢ ⬢ ⬢                 ⬢ ⬢ ⬢ ⬢ ⬢
      ⬢ ⬢ ⬢ ⬢ ⬢                 ⬢ ⬢ ⬢ ⬢ R

Component size: 4          Max component: 1
Better!                    Worse!
```

### 4. Progress

**Concept:** Advancing toward your goal is good

```python
# For Red (connects top to bottom):
progress = row / (dim - 1)  # 0.0 at top, 1.0 at bottom

# For Blue (connects left to right):
progress = col / (dim - 1)  # 0.0 at left, 1.0 at right
```

---

## 🔧 Simple heuristics


```python
def _heuristic_value(self, next_state):
    # Check if game is over
    if next_state.is_done():
        if next_state.get_player_score(self) > 0:
            return 100.0  # We won!
        else:
            return -100.0  # We lost!
    
    # Calculate simple heuristic for non-terminal state
    env = next_state.get_rep().get_env()
    dim = next_state.get_rep().get_dimensions()[0]
    center = (dim - 1) / 2.0
    
    my_score = 0.0
    opp_score = 0.0
    opp_type = "B" if self.piece_type == "R" else "R"
    
    for (i, j), piece in env.items():
        # Distance from center (0.0 to 1.0, higher = closer to center)
        dist_from_center = abs(i - center) + abs(j - center)
        center_bonus = 1.0 / (1.0 + dist_from_center * 0.1)
        
        # Progress toward goal (0.0 to 1.0)
        if piece.get_type() == self.piece_type:
            if self.piece_type == "R":
                progress = i / (dim - 1)  # Top to bottom
            else:
                progress = j / (dim - 1)  # Left to right
            
            my_score += center_bonus + progress * 2.0
        else:
            if opp_type == "R":
                progress = i / (dim - 1)
            else:
                progress = j / (dim - 1)
            
            opp_score += center_bonus + progress * 2.0
    
    return my_score - opp_score
```

---

## 🎯 Strategy Summary

### Opening (First ~10 moves)
- ✅ Play near center (more connections)
- ✅ Start building toward your goal
- ❌ Don't play corners (limited connections)

### Mid-game (Middle ~20 moves)
- ✅ Keep pieces connected
- ✅ Block opponent's best paths
- ✅ Build multiple potential paths

### End-game (Last ~10 moves)
- ✅ Complete shortest path to goal
- ✅ Block opponent's winning moves
- ✅ Force connections

---

## 🐛 Debugging Tips

### Print What Your Heuristic Sees

```python
def _heuristic_value(self, next_state):
    if next_state.is_done():
        result = 100.0 if next_state.get_player_score(self) > 0 else -100.0
        print(f"Terminal state: {result}")
        return result
    
    value = self.calculate_position_value(next_state)
    print(f"Position value: {value:.2f}")
    return value
```

### Visualize Move Values

```python
def debug_moves(self, state):
    """Print heuristic value for each possible move"""
    print("\nEvaluating moves:")
    for action in state.get_possible_heavy_actions():
        next_state = action.get_next_game_state()
        light_action = state.convert_heavy_action_to_light_action(action)
        pos = light_action.data["position"]
        value = self._heuristic_value(next_state)
        print(f"  {pos}: {value:.2f}")
```

---

## 📈 Expected Performance

With proper heuristic:
- **vs Random:** Win rate ~70-80%
- **vs Greedy:** Win rate ~40-50%
- **Rollouts per move:** 500-2000 (depending on board size)

Without proper heuristic (your current code):
- **vs Random:** Win rate ~50% (same as random!)
- **vs Greedy:** Win rate ~20%
- **Rollouts per move:** Fewer (expensive checks slow you down)

---


