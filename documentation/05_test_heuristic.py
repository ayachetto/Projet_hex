"""
Test Script: Verify Your Heuristic is Working

Run this script to check if your heuristic properly differentiates positions.
A working heuristic should return different non-zero values for different positions.

Usage:
    python test_heuristic.py
"""

from board_hex import BoardHex
from game_state_hex import GameStateHex
from player_hex import PlayerHex
from my_player import MyPlayer
from seahorse.game.game_layout.board import Piece

def create_test_board(pieces_dict, dim=11):
    """
    Create a test board with specified pieces.
    
    Args:
        pieces_dict: Dict {(i, j): "R" or "B"}
        dim: Board dimension
    
    Returns:
        GameStateHex
    """
    # Create dummy players
    player1 = PlayerHex("R", "TestRed")
    player2 = PlayerHex("B", "TestBlue")
    
    # Create board
    env = {}
    for pos, piece_type in pieces_dict.items():
        owner = player1 if piece_type == "R" else player2
        env[pos] = Piece(piece_type=piece_type, owner=owner)
    
    board = BoardHex(env=env, dim=dim)
    
    # Create game state
    state = GameStateHex(
        scores={player1.id: 0.0, player2.id: 0.0},
        next_player=player1,
        players=[player1, player2],
        rep=board,
        step=len(pieces_dict)
    )
    
    return state

def test_heuristic():
    """Test if heuristic properly differentiates positions."""
    
    print("=" * 60)
    print("TESTING YOUR HEURISTIC")
    print("=" * 60)
    
    # Create test player
    test_player = MyPlayer("R", "TestPlayer")
    
    # Test 1: Empty board
    print("\n[Test 1] Empty Board")
    print("-" * 40)
    state1 = create_test_board({})
    try:
        value1 = test_player._heuristic_value(state1)
        print(f"✓ Heuristic value: {value1:.3f}")
        if value1 == 0.0:
            print("  → Good: Empty board scores 0.0")
        else:
            print(f"  → Warning: Expected 0.0, got {value1:.3f}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    # Test 2: Single piece at center
    print("\n[Test 2] Single Red Piece at Center (5, 5)")
    print("-" * 40)
    state2 = create_test_board({(5, 5): "R"})
    try:
        value2 = test_player._heuristic_value(state2)
        print(f"✓ Heuristic value: {value2:.3f}")
        if value2 > 0:
            print("  → Good: Center piece has positive value")
        else:
            print(f"  → ERROR: Center piece should have positive value!")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    # Test 3: Single piece at corner
    print("\n[Test 3] Single Red Piece at Corner (0, 0)")
    print("-" * 40)
    state3 = create_test_board({(0, 0): "R"})
    try:
        value3 = test_player._heuristic_value(state3)
        print(f"✓ Heuristic value: {value3:.3f}")
        if value3 > 0 and value3 < value2:
            print("  → Good: Corner < Center (as expected)")
        elif value3 == 0.0:
            print("  → ERROR: Corner piece should have non-zero value!")
            return False
        else:
            print(f"  → Warning: Expected corner ({value3:.3f}) < center ({value2:.3f})")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    # Test 4: Advancing position
    print("\n[Test 4] Red Advancing (pieces forming path)")
    print("-" * 40)
    state4 = create_test_board({
        (0, 5): "R",  # Top
        (2, 5): "R",  # Middle
        (4, 5): "R",  # Advancing
        (6, 5): "R",  # More progress
    })
    try:
        value4 = test_player._heuristic_value(state4)
        print(f"✓ Heuristic value: {value4:.3f}")
        if value4 > value2:
            print("  → Good: Advancing position > single center piece")
        else:
            print(f"  → Warning: Expected advancing ({value4:.3f}) > center ({value2:.3f})")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    # Test 5: Opponent advantage
    print("\n[Test 5] Opponent (Blue) Advantage")
    print("-" * 40)
    state5 = create_test_board({
        (5, 0): "B",  # Blue left
        (5, 3): "B",  # Blue advancing
        (5, 6): "B",  # Blue almost winning
        (5, 9): "B",  # Blue near goal
    })
    try:
        value5 = test_player._heuristic_value(state5)
        print(f"✓ Heuristic value: {value5:.3f}")
        if value5 < 0:
            print("  → Good: Opponent advantage gives negative value")
        else:
            print(f"  → ERROR: Opponent advantage should be negative!")
            return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    # Test 6: Mixed position
    print("\n[Test 6] Mixed Position (Both Players)")
    print("-" * 40)
    state6 = create_test_board({
        (2, 5): "R",
        (4, 5): "R",
        (5, 2): "B",
        (5, 4): "B",
    })
    try:
        value6 = test_player._heuristic_value(state6)
        print(f"✓ Heuristic value: {value6:.3f}")
        print(f"  → Position is slightly {'favoring Red' if value6 > 0 else 'favoring Blue'}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    # Test 7: Check for differentiation
    print("\n[Test 7] Differentiation Check")
    print("-" * 40)
    values = [value1, value2, value3, value4, value5, value6]
    unique_values = len(set(values))
    print(f"Unique values: {unique_values} out of {len(values)}")
    
    if unique_values < 4:
        print("  → ERROR: Heuristic doesn't differentiate well!")
        print("  → Most values are the same - this is the problem!")
        return False
    else:
        print("  → Good: Heuristic returns different values for different positions")
    
    # Test 8: Non-zero check
    print("\n[Test 8] Non-Zero Values Check")
    print("-" * 40)
    non_terminal_values = [value2, value3, value4, value5, value6]
    zero_count = sum(1 for v in non_terminal_values if v == 0.0)
    
    if zero_count > 2:
        print(f"  → ERROR: Too many positions score 0.0 ({zero_count}/{len(non_terminal_values)})")
        print("  → This means your heuristic isn't working!")
        return False
    else:
        print(f"  → Good: Most non-terminal positions have non-zero values")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nAll test values:")
    print(f"  Empty board:           {value1:>8.3f}")
    print(f"  Center piece:          {value2:>8.3f}")
    print(f"  Corner piece:          {value3:>8.3f}")
    print(f"  Advancing position:    {value4:>8.3f}")
    print(f"  Opponent advantage:    {value5:>8.3f}")
    print(f"  Mixed position:        {value6:>8.3f}")
    
    print("\n✅ ALL TESTS PASSED!")
    print("\nYour heuristic appears to be working correctly.")
    print("It differentiates between positions and returns non-zero values.")
    print("\nNext steps:")
    print("  1. Test against random player")
    print("  2. If you're still losing, the heuristic may need tuning")
    print("  3. Consider adding path distance calculation for better accuracy")
    
    return True

def test_broken_heuristic():
    """
    Demonstrate what a broken heuristic looks like.
    (Using game score instead of position features)
    """
    print("\n\n" + "=" * 60)
    print("DEMONSTRATION: BROKEN HEURISTIC")
    print("=" * 60)
    print("\nThis shows what happens when you use game score as heuristic:")
    
    player1 = PlayerHex("R", "Player1")
    player2 = PlayerHex("B", "Player2")
    
    # Various board positions
    positions = [
        ({}, "Empty board"),
        ({(5, 5): "R"}, "Center piece"),
        ({(0, 0): "R"}, "Corner piece"),
        ({(0, 5): "R", (2, 5): "R", (4, 5): "R"}, "Advancing"),
    ]
    
    print("\nUsing state.get_player_score() as heuristic:")
    print("-" * 40)
    
    for pieces, desc in positions:
        env = {}
        for pos, piece_type in pieces.items():
            owner = player1 if piece_type == "R" else player2
            env[pos] = Piece(piece_type=piece_type, owner=owner)
        
        board = BoardHex(env=env, dim=11)
        state = GameStateHex(
            scores={player1.id: 0.0, player2.id: 0.0},
            next_player=player1,
            players=[player1, player2],
            rep=board,
            step=len(pieces)
        )
        
        # Broken heuristic: using game score
        broken_value = state.get_player_score(player1) - state.get_player_score(player2)
        print(f"{desc:20s}: {broken_value:.3f}")
    
    print("\n→ All values are 0.0!")
    print("→ This is why your AI plays randomly!")
    print("→ You MUST use position features, not game score!")

if __name__ == "__main__":
    try:
        success = test_heuristic()
        
        if not success:
            print("\n⚠️  TESTS FAILED!")
            print("\nYour heuristic is not working correctly.")
            print("Common issues:")
            print("  1. Using state.get_player_score() (always returns 0)")
            print("  2. Not evaluating position features")
            print("  3. Returning 0 for all non-terminal states")
            print("\nSee QUICK_REFERENCE.md for a working heuristic template.")
        
        # Show broken heuristic example
        test_broken_heuristic()
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nMake sure you're running this from the project directory.")
        print("The following files should exist:")
        print("  - board_hex.py")
        print("  - game_state_hex.py")
        print("  - player_hex.py")
        print("  - my_player.py")
    except AttributeError as e:
        print(f"\n❌ Error: {e}")
        print("\nYour MyPlayer class might not have the _heuristic_value method.")
        print("Or the method name might be different.")
        print("Check that the method exists and is accessible.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()

