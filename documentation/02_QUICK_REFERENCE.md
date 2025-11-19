# Hex Game AI - Quick Reference Card

## 🗺️ Essential Game State Access

```python
# Get the board
env = current_state.get_rep().get_env()  # Dict: {(i,j): Piece}
dim = current_state.get_rep().get_dimensions()[0]  # e.g., 11

# Check if position is empty
if (i, j) not in env:
    # Empty cell!

# Get piece at position
if (i, j) in env:
    piece_type = env[(i, j)].get_type()  # "R" or "B"

# Get neighbors (6 hexagonal directions)
neighbors = current_state.get_neighbours(i, j)
# Returns: {"top_left": (type, (ni, nj)), ...}
# Types: "EMPTY", "R", "B", "OUTSIDE"

# Iterate all positions
for i in range(dim):
    for j in range(dim):
        # ... check (i, j) ...

# Get empty positions only
empty_cells = list(current_state.get_rep().get_empty())
```

---

## 🏆 Win Conditions

```python
# Red wins: Connect row 0 to row (dim-1)
# Blue wins: Connect column 0 to column (dim-1)

# Check if game is over
if state.is_done():
    if state.get_player_score(self) > 0:
        # We won!
    else:
        # We lost!
```

---

## 💡 Working Heuristic Template

```python
def heuristic(self, state):
    """WORKING heuristic - copy this!"""
    # 1. Handle terminal states
    if state.is_done():
        return 1000.0 if state.get_player_score(self) > 0 else -1000.0
    
    # 2. Extract board info
    env = state.get_rep().get_env()
    dim = state.get_rep().get_dimensions()[0]
    center = (dim - 1) / 2.0
    opp_type = "B" if self.piece_type == "R" else "R"
    
    # 3. Evaluate features
    my_score = 0.0
    opp_score = 0.0
    
    for (i, j), piece in env.items():
        # Center bonus: 1/(1 + distance)
        dist_from_center = abs(i - center) + abs(j - center)
        center_value = 2.0 / (1.0 + dist_from_center * 0.1)
        
        # Progress bonus: how far toward goal (0.0 to 1.0)
        if piece.get_type() == self.piece_type:
            progress = (i if self.piece_type == "R" else j) / (dim - 1)
            my_score += center_value + progress * 2.0
        else:
            progress = (i if opp_type == "R" else j) / (dim - 1)
            opp_score += center_value + progress * 2.0
    
    # 4. Return difference (positive = good for us)
    return my_score - opp_score
```

---

## 📍 Position Values

### What Makes a Position Good?

1. **Center Control** (most important early game)
   - Center: weight = 1.0
   - Edge: weight = 0.3
   - Formula: `1.0 / (1.0 + manhattan_distance_from_center * 0.1)`

2. **Progress** (important mid/late game)
   - Red: advance from row 0 toward row (dim-1)
   - Blue: advance from col 0 toward col (dim-1)
   - Formula: `current_position / (dim - 1)`

3. **Connectivity** (advanced)
   - Connected pieces > isolated pieces
   - Use DFS to find component sizes

4. **Path Distance** (most accurate but slow)
   - Min empty cells needed to win
   - Use Dijkstra's algorithm
   - Only use for root move ordering, NOT rollouts!

---

## 🎲 Hex Neighbors (6 directions)

```
    (i-1, j)  (i-1, j+1)
        ↖  ↗
 (i,j-1) ← ⬢ → (i, j+1)
        ↙  ↘
   (i+1, j-1)  (i+1, j)

# Access via:
neighbors = state.get_neighbours(i, j)
for direction, (type, (ni, nj)) in neighbors.items():
    if type == "EMPTY":
        # Can play here
    elif type == self.piece_type:
        # Our piece (connected)
    elif type in ["R", "B"]:
        # Opponent piece (blocks)
    # type == "OUTSIDE" means off board
```

---

## ⚡ Performance Tips

### For Rollouts (called 1000s of times):
- ✅ Use FAST heuristic (center + progress)
- ✅ Simple arithmetic only
- ❌ NO path searches (Dijkstra)
- ❌ NO "check all opponent moves"
- ❌ NO deep lookahead

### For Root Move Ordering (called ~100 times):
- ✅ Can use SLOW but accurate heuristic
- ✅ Dijkstra path distance
- ✅ Connectivity analysis
- ✅ Deeper evaluation

---

## 🐛 Common Mistakes

### ❌ Mistake 1: Using Game Score as Heuristic
```python
# WRONG: Returns 0.0 until someone wins
heuristic = state.get_player_score(self) - state.get_player_score(opp)
```

### ❌ Mistake 2: Expensive Checks in Rollouts
```python
# WRONG: Way too slow
for action in state.get_possible_heavy_actions():  # 100+ actions
    next_state = action.get_next_game_state()     # 100+ state creations
    if check_something(next_state):               # 100+ recursive checks
        return True
```

### ❌ Mistake 3: No Differentiation
```python
# WRONG: All moves look the same
return 0.0  # for all non-terminal states
```

### ✅ Correct Approach
```python
# RIGHT: Different positions → different values
# Use actual game features (center, progress, connectivity)
return my_features - opponent_features  # Can be any real number
```

---

## 📊 Debugging Your Heuristic

```python
# Test if heuristic works:
def test_heuristic(self):
    # Create two different positions
    state1 = # ... position with center piece
    state2 = # ... position with edge piece
    
    val1 = self.heuristic(state1)
    val2 = self.heuristic(state2)
    
    print(f"Center: {val1}")
    print(f"Edge: {val2}")
    
    # Should see:
    # - Different values (not both 0.0)
    # - Center > Edge usually
    # - Non-zero for non-terminal states
```

---

## 🎯 Strategy Summary

### Opening (moves 1-10)
- ✅ Play near center
- ✅ Start advancing toward goal
- ❌ Avoid corners (limited connectivity)

### Mid-game (moves 11-30)
- ✅ Maintain connected groups
- ✅ Block opponent's main paths
- ✅ Balance offense and defense

### End-game (moves 31+)
- ✅ Complete shortest path
- ✅ Force connections
- ✅ Block opponent wins

---

## ✅ Checklist: Is My Heuristic Working?

- [ ] Returns **non-zero** values for non-terminal states
- [ ] Returns **different** values for different positions
- [ ] **Higher** values for objectively better positions
- [ ] Runs **fast** (no expensive searches in rollouts)
- [ ] Uses **game features** (not just piece counts)
- [ ] Handles **terminal states** (returns ±1000 when someone won)

If all checked: Your heuristic should work! 🎉

