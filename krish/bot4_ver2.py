"""
Bot4 Ver2: Bot4 variant with max auction bids.

Behavior change from bot4:
- Auction: always bid maximum legal amount (entire stack).
"""
from pkbot.runner import parse_args, run_bot

import bot4


class Player(bot4.Player):
    def _play_auction(self, game_info, state, t0):
        # Keep auction tracking fields consistent with base implementation.
        self.hand_auction_snapshot = (state.my_chips, state.opp_chips)
        self.hand_my_bid = int(state.my_chips)
        return self._safe_bid(state, state.my_chips)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
