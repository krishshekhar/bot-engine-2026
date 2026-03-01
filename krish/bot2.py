"""
Bot2: see-cards then jam bot.

Strategy:
- Pre-flop: passive to reach flop cheaply.
- Auction: bid 0 (legal and fast).
- Flop/Turn/River: if raise is legal, raise to maximum legal amount.
  Otherwise call if possible, else check/fold fallback.
"""
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState


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
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionCall):
                return ActionCall()
            return ActionFold()

        # Post-flop streets: jam whenever legal.
        if current_state.can_act(ActionRaise):
            min_raise, max_raise = current_state.raise_bounds
            amount = max(min_raise, max_raise)
            return ActionRaise(int(amount))

        if current_state.can_act(ActionCall):
            return ActionCall()
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
