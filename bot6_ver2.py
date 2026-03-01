"""
Standalone max-auction + jam-after-auction-win bot.

Strategy:
- Auction: always bid max legal amount.
- Post-auction betting streets: if we won the auction, jam whenever legal.
- Otherwise use simple safe fallback actions.
"""

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState


class Player(BaseBot):
    def __init__(self) -> None:
        self.auction_result_known = False
        self.auction_won = False
        self.hand_auction_snapshot = None

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.auction_result_known = False
        self.auction_won = False
        self.hand_auction_snapshot = None

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        return

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        if current_state.street == "auction":
            self.hand_auction_snapshot = (current_state.my_chips, current_state.opp_chips)
            if current_state.can_act(ActionBid):
                return ActionBid(int(current_state.my_chips))
            return self._fallback(current_state)

        self._resolve_auction_outcome(current_state)
        if self._should_force_jam(current_state):
            return self._jam_action(current_state)
        return self._fallback(current_state)

    def _resolve_auction_outcome(self, state: PokerState) -> None:
        if self.auction_result_known:
            return
        if state.street not in ("flop", "turn", "river"):
            return

        if self.hand_auction_snapshot is None:
            self.auction_won = False
            self.auction_result_known = True
            return

        before_my, before_opp = self.hand_auction_snapshot
        d_my = max(0, before_my - state.my_chips)
        d_opp = max(0, before_opp - state.opp_chips)

        # Auction win: only our stack decreases after auction resolution.
        self.auction_won = (d_my > 0 and d_opp == 0)
        self.auction_result_known = True

    def _should_force_jam(self, state: PokerState) -> bool:
        return state.street in ("flop", "turn", "river") and self.auction_won

    def _jam_action(self, state: PokerState):
        if state.can_act(ActionRaise):
            _, hi = state.raise_bounds
            return ActionRaise(int(hi))
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()

    def _fallback(self, state: PokerState):
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionBid):
            return ActionBid(0)
        return ActionFold()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
