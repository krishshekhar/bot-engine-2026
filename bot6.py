"""
Bot6: pure legal all-in pressure bot.

Behavior:
- Auction: bids entire stack.
- Any betting street: if raise is legal, raise to max legal amount.
- Otherwise call if possible, else check, else fold.
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
                return ActionBid(int(current_state.my_chips))
            if current_state.can_act(ActionCheck):
                return ActionCheck()
            if current_state.can_act(ActionCall):
                return ActionCall()
            return ActionFold()

        if current_state.can_act(ActionRaise):
            _, max_raise = current_state.raise_bounds
            return ActionRaise(int(max_raise))

        if current_state.can_act(ActionCall):
            return ActionCall()
        if current_state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
