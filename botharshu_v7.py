"""
v7: Simple, risk-aware bot with:
- Auction: always bids at least 1 and usually tries to win the auction.
- Preflop: not aggressive; folds to high raises unless holding a pocket pair.
- Postflop: binary strategy:
    * If equity is very high, play hard (big raise).
    * Otherwise, fold immediately when facing big bets; rarely bluff.
"""

from collections import deque
import random
import time

import eval7

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState


RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i + 2 for i, r in enumerate(RANK_ORDER)}


class Player(BaseBot):
    def __init__(self) -> None:
        self.rng = random.Random(2027)
        self.round_num = 0

        # Auction model (simple tracking, mainly for prediction).
        self.opp_bid_exact_samples = deque(maxlen=260)
        self.opp_bid_lower_bounds = deque(maxlen=260)

        # Equity cache
        self.equity_cache = {}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.round_num = game_info.round_num
        self.rng.seed((self.round_num * 7919) + (17 if current_state.is_bb else 11))

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        return

    def get_move(
        self,
        game_info: GameInfo,
        current_state: PokerState,
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        t0 = time.perf_counter()

        if current_state.street == "auction":
            bid = self._choose_auction_bid(game_info, current_state, t0)
            return self._safe_bid(current_state, bid)

        if current_state.street == "pre-flop":
            return self._play_preflop(game_info, current_state)

        return self._play_postflop(game_info, current_state, t0)

    # ---------- Auction ----------

    def _predict_opp_bid(self, opp_stack: int, pot: int) -> int:
        if self.opp_bid_exact_samples:
            mean_exact = sum(self.opp_bid_exact_samples) / float(len(self.opp_bid_exact_samples))
        else:
            mean_exact = 0.10 * min(opp_stack, pot)

        if self.opp_bid_lower_bounds:
            lb = sum(self.opp_bid_lower_bounds) / float(len(self.opp_bid_lower_bounds))
            pred = 0.7 * mean_exact + 0.3 * lb
        else:
            pred = mean_exact

        if self.opp_bid_lower_bounds:
            lb_hi = max(self.opp_bid_lower_bounds)
            pred = max(pred, 0.5 * lb_hi)

        pred *= 1.05
        return int(self._clip(pred, 1, opp_stack))

    def _choose_auction_bid(self, game_info: GameInfo, state: PokerState, t0: float) -> int:
        """
        Always bid at least 1, and generally aim to slightly outbid
        the opponent prediction so we win the auction most of the time.
        """
        pot = max(1, state.pot)
        eff = max(1, min(state.my_chips, state.opp_chips))

        pred = self._predict_opp_bid(state.opp_chips, pot)
        target = max(pred + 1, int(0.02 * eff), 1)

        # Do not risk entire stack; cap to ~25% of effective or 2x pot.
        hard_cap = int(min(state.my_chips, max(1, min(0.25 * eff, 2.0 * pot))))
        bid = int(self._clip(target, 1, hard_cap))

        # Add some noise so we aren't perfectly predictable.
        sigma = max(1.0, 0.15 * max(1.0, float(bid)))
        noisy = bid + self.rng.gauss(0.0, sigma)
        bid = int(self._clip(noisy, 1, hard_cap))

        return bid

    # ---------- Preflop ----------

    def _has_pair(self, hand: list[str]) -> bool:
        return hand[0][0] == hand[1][0]

    def _play_preflop(self, game_info: GameInfo, state: PokerState):
        """
        - Mostly call/check with reasonable hands.
        - Fold to high preflop raises unless we have a pocket pair.
        - Do not 3-bet/4-bet aggressively.
        """
        can_raise = state.can_act(ActionRaise)
        can_call = state.can_act(ActionCall)
        can_check = state.can_act(ActionCheck)

        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        my_stack = max(1, state.my_chips)

        pair = self._has_pair(state.my_hand)

        # Define "high raise" preflop as a big chunk of our stack or big vs pot.
        high_raise = (call_cost > 0.18 * my_stack) or (call_cost > 2.5 * pot)

        # If facing all-in / huge raise:
        if can_call and call_cost >= state.my_chips:
            if pair:
                return ActionCall()
            # Non-pairs: just fold.
            return ActionFold() if state.can_act(ActionFold) else ActionCall()

        # Fold to high raises preflop unless we have a pair.
        if high_raise and not pair:
            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCall()  # safety fallback

        # For small raises / limped pots:
        if pair:
            # With pairs, we are willing to call small raises.
            if can_call and call_cost > 0:
                return ActionCall()
            if can_check:
                return ActionCheck()
            if can_call:
                return ActionCall()
            return ActionFold()

        # Non-pair hands: be passive preflop.
        if call_cost == 0:
            # Just check, do not open-raise aggressively.
            if can_check:
                return ActionCheck()
            if can_call:
                return ActionCall()
            return ActionFold()

        # Call only small raises with non-pairs.
        if can_call and call_cost <= 0.08 * my_stack:
            return ActionCall()

        if can_check:
            return ActionCheck()
        if can_call:
            return ActionCall()
        return ActionFold()

    # ---------- Postflop ----------

    def _play_postflop(self, game_info: GameInfo, state: PokerState, t0: float):
        board = state.board
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        my_stack = max(1, state.my_chips)

        # Simple equity estimate ignoring revealed card for now.
        iters = self._choose_mc_iters(game_info.time_bank, state.street)
        eq = self._estimate_equity(state.my_hand, board, [], iters, t0, game_info.time_bank)

        # Heuristics:
        # - "Very high" equity: eq >= 0.80  (roughly trips+ / strong combo)
        # - "Very low" equity: eq <= 0.40  (unlikely to win)
        very_strong = eq >= 0.80
        very_weak = eq <= 0.40

        # Facing a bet:
        if call_cost > 0:
            rel_price = call_cost / float(pot + call_cost)
            stack_commit = call_cost / float(my_stack)

            # If equity is very low, just fold to any bet.
            if very_weak:
                if state.can_act(ActionFold):
                    return ActionFold()
                if state.can_act(ActionCheck):
                    return ActionCheck()
                return self._fallback_action(state)

            # If equity is not very strong and the bet is large, fold.
            if not very_strong and (rel_price > 0.40 or stack_commit > 0.35):
                if state.can_act(ActionFold):
                    return ActionFold()
                if state.can_act(ActionCheck):
                    return ActionCheck()
                return self._fallback_action(state)

            # With very strong equity, raise big to get value.
            if very_strong and state.can_act(ActionRaise):
                min_r, max_r = state.raise_bounds
                # Aim for a large raise: between 60-80% of pot, capped by max_r.
                target = int(self._clip(0.7 * pot, min_r, max_r))
                return self._safe_raise_or_fallback(state, target)

            # Medium equity facing small bets: just call.
            if state.can_act(ActionCall):
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return self._fallback_action(state)

        # Checked to us:
        if very_strong and state.can_act(ActionRaise):
            min_r, max_r = state.raise_bounds
            target = int(self._clip(0.7 * pot, min_r, max_r))
            return self._safe_raise_or_fallback(state, target)

        # Otherwise, take the free card / showdown.
        if state.can_act(ActionCheck):
            return ActionCheck()
        return self._fallback_action(state)

    # ---------- Helpers ----------

    def _safe_raise_or_fallback(self, state: PokerState, amount: int):
        if state.can_act(ActionRaise):
            min_r, max_r = state.raise_bounds
            amt = int(self._clip(amount, min_r, max_r))
            if min_r <= amt <= max_r:
                return ActionRaise(amt)
        return self._fallback_action(state)

    def _safe_bid(self, state: PokerState, amount: int):
        if state.can_act(ActionBid):
            # Always bid at least 1 if we have chips.
            if state.my_chips <= 0:
                return self._fallback_action(state)
            amt = int(self._clip(amount, 1, state.my_chips))
            return ActionBid(amt)
        return self._fallback_action(state)

    def _fallback_action(self, state: PokerState):
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionBid):
            return ActionBid(1)
        return ActionFold()

    # ---------- Equity estimation ----------

    def _estimate_equity(
        self,
        my_hand: list[str],
        board: list[str],
        opp_revealed: list[str],
        iters: int,
        t0: float,
        time_bank: float,
    ) -> float:
        key = (tuple(sorted(my_hand)), tuple(board), len(board), iters // 40)
        if key in self.equity_cache:
            return self.equity_cache[key]

        known_cards = my_hand + board + opp_revealed
        if len(set(known_cards)) != len(known_cards):
            return 0.5

        my_cards = [eval7.Card(c) for c in my_hand]
        board_cards = [eval7.Card(c) for c in board]

        all_cards = [eval7.Card(r + s) for r in RANK_ORDER for s in "shdc"]
        dead = set(my_cards + board_cards)
        rem = [c for c in all_cards if c not in dead]

        need_board = 5 - len(board_cards)
        wins = 0.0
        n = 0
        max_runtime = self._per_decision_time_budget(time_bank)

        for _ in range(max(20, iters)):
            if time.perf_counter() - t0 > max_runtime:
                break

            drawn = self.rng.sample(rem, need_board + 2)
            opp_cards = [drawn[0], drawn[1]]
            board_draw = drawn[2:]

            full_board = board_cards + board_draw
            my_score = eval7.evaluate(my_cards + full_board)
            opp_score = eval7.evaluate(opp_cards + full_board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
            n += 1

        eq = wins / n if n > 0 else 0.5
        if len(self.equity_cache) > 4000:
            self.equity_cache.clear()
        self.equity_cache[key] = eq
        return eq

    def _choose_mc_iters(self, time_bank: float, street: str) -> int:
        if time_bank < 2.5:
            base = 60
        elif time_bank < 6.0:
            base = 100
        else:
            base = 150

        if street == "river":
            return int(base * 0.7)
        if street == "turn":
            return int(base * 0.85)
        if street == "flop":
            return int(base * 1.1)
        if street == "auction":
            return int(base * 0.8)
        return base

    def _per_decision_time_budget(self, time_bank: float) -> float:
        if time_bank < 1.5:
            return 0.012
        if time_bank < 4.0:
            return 0.020
        if time_bank < 8.0:
            return 0.030
        return 0.050

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x


if __name__ == "__main__":
    run_bot(Player(), parse_args())

