"""
Proxy bot approximating POCKET_D_ACE behavior from logs:
- Preflop: limp-heavy, no voluntary raises.
- Versus all-in: calls with stronger bucket, folds weak trash.
- Auction: always 0.
- Postflop: check/call conservative fallback.
"""
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState

RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i + 2 for i, r in enumerate(RANK_ORDER)}


class Player(BaseBot):
    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        return None

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        return None

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        if current_state.street == "auction":
            if current_state.can_act(ActionBid):
                return ActionBid(0)
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionCall):
                return ActionCall()
            return ActionFold()

        if current_state.street == "pre-flop":
            if current_state.can_act(ActionCall):
                # Limp from SB, or decide versus jam-like pressure.
                if current_state.cost_to_call <= 40:
                    return ActionCall()
                return ActionCall() if self._should_call_preflop_allin(current_state) else ActionFold()
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold()

        if current_state.can_act(ActionCheck):
            return ActionCheck()
        if current_state.can_act(ActionCall):
            return ActionCall()
        if current_state.can_act(ActionRaise):
            lo, _ = current_state.raise_bounds
            return ActionRaise(int(lo))
        return ActionFold()

    def _should_call_preflop_allin(self, state: PokerState) -> bool:
        c1, c2 = state.my_hand
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        v1 = RANK_TO_INT[r1]
        v2 = RANK_TO_INT[r2]
        hi = max(v1, v2)
        lo = min(v1, v2)
        suited = s1 == s2
        pair = v1 == v2
        gap = abs(v1 - v2)

        if pair and hi >= 7:
            return True
        if pair and hi >= 4 and state.cost_to_call <= 1800:
            return True
        if hi >= 13 and lo >= 10:
            return True
        if hi == 14 and suited:
            return True
        if suited and hi >= 11 and gap <= 2:
            return True
        if suited and hi >= 10 and gap == 1 and state.cost_to_call <= 1600:
            return True

        req = state.cost_to_call / float(max(1, state.pot + state.cost_to_call))
        return req <= 0.22 and hi >= 11


if __name__ == "__main__":
    run_bot(Player(), parse_args())
