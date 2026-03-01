"""
Bot6 Ver2: max-auction + conditional jam wrapper.

Behavior:
- Always bid max legal amount in auction to strongly contest info.
- Uses Bot5 as the normal strategy engine.
- After auction resolution:
  - If our estimated winning chance is >= 50%, keep normal Bot5 play.
  - If estimated winning chance is < 50%, force max-pressure all-in when legal.
"""

import time

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState

import bot5


class Player(bot5.Player):
    def __init__(self) -> None:
        super().__init__()
        self.auction_result_known = False
        self.auction_won_or_shared = False

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        super().on_hand_start(game_info, current_state)
        self.auction_result_known = False
        self.auction_won_or_shared = False

    def get_move(self, game_info: GameInfo, current_state: PokerState):
        # Force a max legal auction bid every hand.
        if current_state.street == "auction":
            self.hand_auction_snapshot = (current_state.my_chips, current_state.opp_chips)
            self.hand_auction_processed = False
            self.hand_my_bid = int(current_state.my_chips)
            if current_state.can_act(ActionBid):
                return ActionBid(int(current_state.my_chips))
            return self._fallback(current_state)

        # Resolve whether we won/shared auction info once betting streets start.
        self._resolve_auction_outcome(current_state)

        # If we got the info edge but still project behind, switch to max pressure.
        if self._should_force_jam(game_info, current_state):
            return self._jam_action(current_state)

        # Otherwise play normal strong policy (Bot5).
        return super().get_move(game_info, current_state)

    def _resolve_auction_outcome(self, state: PokerState) -> None:
        if self.auction_result_known:
            return
        if state.street not in ("flop", "turn", "river"):
            return
        if self.hand_auction_snapshot is None:
            self.auction_result_known = True
            self.auction_won_or_shared = False
            return

        before_my, before_opp = self.hand_auction_snapshot
        d_my = max(0, before_my - state.my_chips)
        d_opp = max(0, before_opp - state.opp_chips)

        # d_my>0,d_opp==0 => we won auction.
        # d_my>0,d_opp>0 => tie auction (shared info), still treated as info received.
        self.auction_won_or_shared = (d_my > 0)
        self.auction_result_known = True

    def _should_force_jam(self, game_info: GameInfo, state: PokerState) -> bool:
        if state.street not in ("flop", "turn", "river"):
            return False
        if not self.auction_won_or_shared:
            return False
        if game_info.time_bank < 0.6:
            return False

        t0 = time.perf_counter()
        eq = self._estimate_equity(
            state.my_hand,
            state.board,
            state.opp_revealed_cards,
            48,
            t0,
            game_info.time_bank,
        )
        return eq < 0.50

    def _jam_action(self, state: PokerState):
        if state.can_act(ActionRaise):
            _, hi = state.raise_bounds
            return ActionRaise(int(hi))
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
