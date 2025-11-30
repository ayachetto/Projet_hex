from seahorse.game.game_state import GameState
from typing import Dict


class HexPlayerDebug:
    
    def __init__(self, player, helper):
        self.player = player
        self.helper = helper
    
    def analyze_position(self, state: GameState) -> Dict:
        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        
        my_path = self.helper.shortest_path_distance(state, self.player.piece_type, consider_bridges=False)
        my_virtual = self.helper.virtual_connection_distance(state, self.player.piece_type)
        opp_path = self.helper.shortest_path_distance(state, self.player.opp_type, consider_bridges=False)
        opp_virtual = self.helper.virtual_connection_distance(state, self.player.opp_type)
        
        my_bridges = self.helper.find_bridges(state, self.player.piece_type)
        opp_bridges = self.helper.find_bridges(state, self.player.opp_type)
        
        threatened = self.helper.find_threatened_bridges(state)
        urgent_responses = self.helper.find_urgent_bridge_responses(state)
        
        avoid_positions = self.helper.find_own_complete_bridge_carriers(state)
        
        cut_points = self.helper.find_cut_points(state, self.player.opp_type)
        bridge_opps = self.helper.find_bridge_opportunities(state, self.player.piece_type)
        
        bridge_opps = [(pos, val) for pos, val in bridge_opps if pos not in avoid_positions]
        
        center = (dim - 1) / 2.0
        my_center = sum(1.0 / (1.0 + abs(i - center) + abs(j - center))
                       for (i, j), p in env.items() if p.get_type() == self.player.piece_type)
        opp_center = sum(1.0 / (1.0 + abs(i - center) + abs(j - center))
                        for (i, j), p in env.items() if p.get_type() == self.player.opp_type)
        
        dynamic_center_weight = self.helper.get_dynamic_center_weight(state)
        game_phase = self.helper.get_game_phase(state)
        
        current_step = state.get_step()
        max_steps = state.max_step
        game_progress = current_step / max_steps
        
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
            'heuristic_value': self.helper.accurate_heuristic(state),
        }
    
    def print_bridge_analysis(self, state: GameState) -> None:
        analysis = self.analyze_position(state)
        
        goal_desc = "TOP(row 0)→BOTTOM(row N-1)" if self.player.piece_type == "R" else "LEFT(col 0)→RIGHT(col N-1)"
        
        print(f"\n{'='*50}")
        print(f"POSITION ANALYSIS - {self.player.piece_type} to move")
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
        
        if analysis['urgent_responses']:
            print(f"\n🚨 URGENT - MUST DEFEND BRIDGES:")
            for pos, urgency in analysis['urgent_responses']:
                print(f"   PLAY {pos} (urgency: {urgency:.2f})")
        
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
        print(f"   (decays from {self.player.W_CENTER_BASE} to {self.player.W_CENTER_BASE * self.player.W_CENTER_DECAY:.2f})")
        
        print(f"\n🎮 GAME PHASE: {analysis['game_phase'].upper()}")
        print(f"   Progress: {analysis['game_progress']*100:.1f}%")
        
        print(f"\n📊 HEURISTIC VALUE: {analysis['heuristic_value']:.2f}")
        
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

