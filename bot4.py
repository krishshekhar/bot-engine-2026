"""
Bot4: Hybrid GTO baseline + exploit adaptation + lightweight neural scorer.

Design goals:
- Keep legal-action safety and fast runtime.
- Start from stable range-based thresholds (GTO-ish baseline).
- Adapt to opponent behavior (bet sizing, fold-to-raise, auction style).
- Use a tiny fixed-weight neural evaluator for action pressure.
"""
from collections import deque
import math
import random
import time
from typing import Union

try:
    import eval7
except Exception:
    eval7 = None

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState


RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i + 2 for i, r in enumerate(RANK_ORDER)}


class Player(BaseBot):
    def __init__(self) -> None:
        self.rng = random.Random(404)
        self.round_num = 0

        # Opponent behavior model
        self.our_raise_opps = 0
        self.opp_fold_to_raise_hits = 0
        self.opp_post_bet_spots = 0
        self.opp_small_stab_hits = 0
        self.opp_huge_bet_hits = 0
        self.opp_small_pre_hits = 0
        self.opp_massive_pre_hits = 0
        self.opp_pre_raise_spots = 0
        self.opp_pre_rejam_spots = 0
        self.opp_pre_rejam_hits = 0

        # Auction behavior model
        self.opp_bid_exact = deque(maxlen=260)
        self.opp_bid_lb = deque(maxlen=260)
        self.auction_total = 0
        self.auction_my_wins = 0
        self.auction_opp_wins = 0
        self.auction_high_opp_wins = 0
        self.hand_my_bid = None
        self.hand_auction_snapshot = None
        self.hand_auction_processed = False
        self.hand_i_raised = False
        self.hand_pre_raised = False

        self.equity_cache = {}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.round_num = game_info.round_num
        self.rng.seed((self.round_num * 7919) + (19 if current_state.is_bb else 13))
        self.hand_my_bid = None
        self.hand_auction_snapshot = None
        self.hand_auction_processed = False
        self.hand_i_raised = False
        self.hand_pre_raised = False

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        if self.hand_i_raised:
            self.our_raise_opps += 1
            if len(current_state.opp_revealed_cards) < 2 and current_state.payoff > 0:
                self.opp_fold_to_raise_hits += 1

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> Union[ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid]:
        t0 = time.perf_counter()
        self._update_models(current_state)

        if game_info.time_bank < 0.9:
            return self._panic_action(current_state)

        if current_state.street == "auction":
            return self._play_auction(game_info, current_state, t0)
        if current_state.street == "pre-flop":
            return self._play_preflop(game_info, current_state, t0)
        return self._play_postflop(game_info, current_state, t0)

    # ---------- Street logic ----------

    def _play_preflop(self, game_info: GameInfo, state: PokerState, t0: float):
        key = self._preflop_key(state.my_hand[0], state.my_hand[1])
        tier = self._preflop_tier(key)

        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(max(1, pot + call_cost))

        small_pre = self._rate(self.opp_small_pre_hits, self.opp_pre_raise_spots, 0.25)
        massive_pre = self._rate(self.opp_massive_pre_hits, self.opp_pre_raise_spots, 0.08)
        rejam_rate = self._rate(self.opp_pre_rejam_hits, self.opp_pre_rejam_spots, 0.10)
        fold_rate = self._opp_fold_rate()

        if call_cost >= state.my_chips and state.can_act(ActionCall):
            # Baseline all-in defense with exploit correction.
            if tier >= 2 or pot_odds < (0.25 if massive_pre < 0.18 else 0.20):
                return ActionCall()
            return ActionFold() if state.can_act(ActionFold) else ActionCall()

        # Re-raise trap brake: after we opened, avoid paying too much with weak tiers.
        if self.hand_pre_raised and call_cost > 0 and rejam_rate > 0.18:
            price = call_cost / float(max(1, state.my_chips))
            if tier <= 1 and price > 0.08:
                return ActionFold() if state.can_act(ActionFold) else ActionCall()
            if tier == 2 and price > 0.20 and pot_odds > 0.24:
                return ActionFold() if state.can_act(ActionFold) else ActionCall()

        # GTO-ish baseline mix
        if tier == 3:
            raise_p, call_p = 0.90, 0.08
        elif tier == 2:
            raise_p, call_p = 0.72, 0.20
        elif tier == 1:
            raise_p, call_p = 0.48, 0.34
        else:
            raise_p, call_p = 0.26, 0.30

        # Exploit adjustments
        if fold_rate > 0.5:
            raise_p += 0.10
        if small_pre > 0.45 and call_cost <= 100:
            if tier <= 1:
                raise_p -= 0.12
                call_p += 0.10
        if massive_pre > 0.16 and call_cost > 0.22 * state.my_chips:
            if tier <= 1:
                raise_p *= 0.65
                call_p *= 0.70
        if rejam_rate > 0.22 and tier <= 1:
            # Opponent attacks our opens with re-jams: avoid spewy weak open-raises.
            raise_p *= 0.56
            call_p *= 1.08
        if rejam_rate > 0.30 and call_cost <= 100 and tier <= 1:
            raise_p = min(raise_p, 0.14 if tier == 0 else 0.30)
        if call_cost == 0 and tier == 0:
            raise_p = min(raise_p, 0.18)
        if self.hand_pre_raised and call_cost > 0 and rejam_rate > 0.20 and tier <= 1:
            raise_p *= 0.20
            call_p *= 0.85

        # Neural pressure score modifies aggression smoothly.
        score = self._neural_score(
            eq_proxy=self._tier_to_eq_proxy(tier),
            pot_odds=pot_odds,
            fold_rate=fold_rate,
            pressure=(call_cost / float(max(1, state.my_chips))),
            wetness=0.0,
        )
        raise_p += 0.12 * (score - 0.5)
        call_p -= 0.08 * (score - 0.5)

        raise_p = self._clip(raise_p, 0.0, 1.0)
        call_p = self._clip(call_p, 0.0, 0.95)

        roll = self.rng.random()
        if state.can_act(ActionRaise) and roll < raise_p:
            amt = self._preflop_raise_amount(state, tier, small_pre, rejam_rate)
            return self._safe_raise(state, amt)
        if state.can_act(ActionCall) and roll < raise_p + call_p:
            return ActionCall()
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall) and tier >= 1:
            return ActionCall()
        return ActionFold()

    def _play_postflop(self, game_info: GameInfo, state: PokerState, t0: float):
        eq = self._estimate_equity(
            state.my_hand, state.board, state.opp_revealed_cards,
            self._mc_iters(game_info.time_bank, state.street), t0, game_info.time_bank
        )
        wet = self._board_wetness(state.board)
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(max(1, pot + call_cost))
        rel_price = call_cost / float(max(1, pot))
        fold_rate = self._opp_fold_rate()

        score = self._neural_score(
            eq_proxy=eq,
            pot_odds=pot_odds,
            fold_rate=fold_rate,
            pressure=rel_price,
            wetness=wet / 3.0,
        )

        if call_cost > 0:
            raise_th = 0.60 + 0.05 * max(0.0, rel_price - 0.5)
            call_th = pot_odds + 0.03 + (0.04 if rel_price > 0.85 else 0.0)
            if self._rate(self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18) > 0.22:
                call_th += 0.04

            if state.can_act(ActionRaise) and eq > raise_th and score > 0.48:
                amt = self._post_raise_amount(state, eq, rel_price, wet)
                return self._safe_raise(state, amt)
            if state.can_act(ActionCall) and eq >= call_th:
                return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
            return self._fallback_action(state)

        # Checked to us
        if state.can_act(ActionRaise):
            value_bet = eq > (0.55 if state.street == "flop" else 0.58)
            exploit_bet = fold_rate > 0.44 and score > 0.54
            if value_bet or exploit_bet:
                amt = self._post_raise_amount(state, eq, 0.0, wet)
                return self._safe_raise(state, amt)

        if state.can_act(ActionCheck):
            return ActionCheck()
        return self._fallback_action(state)

    def _play_auction(self, game_info: GameInfo, state: PokerState, t0: float):
        self.hand_auction_snapshot = (state.my_chips, state.opp_chips)
        pot = max(1, state.pot)
        pred = self._predict_opp_bid(state.opp_chips, pot)
        loss_rate = self._auction_loss_rate()
        high_rate = self._opp_high_bid_win_rate()

        eq = self._estimate_equity(state.my_hand, state.board, [], self._mc_iters(game_info.time_bank, "auction"), t0, game_info.time_bank)
        info_value = self._clip(0.22 - abs(eq - 0.5), 0.02, 0.22) * pot

        # Regime selection by opponent bidding style.
        if self._is_micro_bidder(pred, pot):
            target = pred + (2 if eq < 0.68 else 1)
        elif self._is_extreme_bidder(pred, pot):
            # Contest info against persistent overbidders; avoid free revelation.
            if eq < 0.62:
                target = int(max(0.52 * pred, 0.26 * pot))
            elif eq < 0.78:
                target = int(max(0.42 * pred, 0.20 * pot))
            else:
                target = int(max(0.30 * pred, 0.14 * pot))
        elif self._is_high_mid_anchor_bidder(pred):
            if eq > 0.72 and self.rng.random() < 0.35:
                target = 0
            elif 0.44 <= eq <= 0.62 and self.rng.random() < 0.24:
                target = pred + int(self.rng.uniform(-3, 6))
            else:
                target = min(pred // 2, int(0.16 * pot))
        else:
            target = int(0.54 * info_value + 0.46 * pred)

        # If we're losing auctions often, increase contest frequency and bid floor.
        if self.auction_total >= 8:
            if loss_rate > 0.62:
                target = int(max(target, 0.72 * pred + 0.10 * pot))
            elif loss_rate > 0.54:
                target = int(max(target, 0.62 * pred + 0.06 * pot))
            if high_rate > 0.35:
                target = int(max(target, 0.78 * pred))

        # Strong contest in middling-equity regions where info value is highest.
        if 0.42 <= eq <= 0.74 and pred >= max(160, int(0.18 * max(1, state.opp_chips))):
            target = int(max(target, pred + max(8, int(0.06 * pot))))

        # Neural smoothing + bankroll preservation.
        n = self._neural_score(eq, 0.0, self._opp_fold_rate(), 0.0, self._board_wetness(state.board) / 3.0)
        target = int(target * (0.80 + 0.35 * n))
        bid = int(self._clip(target + self.rng.randint(-6, 6), 0, state.my_chips))

        # Additional floor against persistent high bidders.
        if self.auction_total >= 10 and (loss_rate > 0.58 or high_rate > 0.30):
            floor = int(self._clip(max(16, 0.35 * pred), 0, state.my_chips))
            bid = max(bid, floor)

        if self.rng.random() < 0.06 and not self._is_extreme_bidder(pred, pot):
            bid = 0

        self.hand_my_bid = bid
        return self._safe_bid(state, bid)

    # ---------- Models ----------

    def _update_models(self, state: PokerState) -> None:
        if state.street == "pre-flop" and state.cost_to_call > 0:
            self.opp_pre_raise_spots += 1
            if state.opp_wager <= 120 or state.cost_to_call <= 80:
                self.opp_small_pre_hits += 1
            if state.opp_wager >= 900 or state.cost_to_call >= 850:
                self.opp_massive_pre_hits += 1
            if self.hand_pre_raised:
                self.opp_pre_rejam_spots += 1
                if state.opp_wager >= max(180, int(0.22 * max(1, state.my_chips))):
                    self.opp_pre_rejam_hits += 1

        if state.street in ("flop", "turn", "river") and state.cost_to_call > 0:
            self.opp_post_bet_spots += 1
            rel = state.cost_to_call / float(max(1, state.pot))
            if rel <= 0.18:
                self.opp_small_stab_hits += 1
            if rel >= 0.90:
                self.opp_huge_bet_hits += 1

        if (
            state.street == "flop"
            and self.hand_my_bid is not None
            and self.hand_auction_snapshot is not None
            and not self.hand_auction_processed
        ):
            before_my, before_opp = self.hand_auction_snapshot
            d_my = max(0, before_my - state.my_chips)
            d_opp = max(0, before_opp - state.opp_chips)
            if d_my > 0 and d_opp == 0:
                self.auction_total += 1
                self.auction_my_wins += 1
                self.opp_bid_exact.append(d_my)
            elif d_opp > 0 and d_my == 0:
                self.auction_total += 1
                self.auction_opp_wins += 1
                if d_opp >= 120:
                    self.auction_high_opp_wins += 1
                self.opp_bid_lb.append(max(1, self.hand_my_bid + 1))
            elif d_my > 0 and d_opp > 0:
                self.auction_total += 1
                # Tie auction: both reveal and both pay own bids. Keep neutral in win/loss.
                self.opp_bid_exact.append(d_opp)
            self.hand_auction_processed = True

    # ---------- Action safety ----------

    def _safe_raise(self, state: PokerState, amount: int):
        if state.can_act(ActionRaise):
            lo, hi = state.raise_bounds
            amt = int(self._clip(amount, lo, hi))
            self.hand_i_raised = True
            if state.street == "pre-flop":
                self.hand_pre_raised = True
            return ActionRaise(amt)
        return self._fallback_action(state)

    def _safe_bid(self, state: PokerState, amount: int):
        if state.can_act(ActionBid):
            return ActionBid(int(self._clip(amount, 0, state.my_chips)))
        return self._fallback_action(state)

    def _fallback_action(self, state: PokerState):
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionBid):
            return ActionBid(0)
        return ActionFold()

    def _panic_action(self, state: PokerState):
        if state.street == "auction":
            return self._safe_bid(state, min(4, state.my_chips))
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        return ActionFold()

    # ---------- Helpers ----------

    def _preflop_key(self, c1: str, c2: str) -> str:
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        if r1 == r2:
            return r1 + r2
        hi, lo = (r1, r2) if RANK_TO_INT[r1] >= RANK_TO_INT[r2] else (r2, r1)
        return hi + lo + ("s" if s1 == s2 else "o")

    def _preflop_tier(self, key: str) -> int:
        if key in ("AA", "KK", "QQ", "JJ", "AKs", "AQs", "AKo"):
            return 3
        if key in ("TT", "99", "88", "AJs", "ATs", "KQs", "KJs", "QJs", "AQo", "AJo", "KQo"):
            return 2
        if key in ("77", "66", "55", "44", "33", "22", "JTs", "T9s", "98s", "87s", "A9s", "KTs", "QTs", "ATo", "KJo", "QJo"):
            return 1
        return 0

    def _tier_to_eq_proxy(self, tier: int) -> float:
        return [0.42, 0.50, 0.60, 0.70][tier]

    def _preflop_raise_amount(self, state: PokerState, tier: int, small_pre: float, rejam_rate: float) -> int:
        lo, hi = state.raise_bounds
        span = max(0, hi - lo)
        frac = 0.90 if tier >= 3 else 0.72 if tier == 2 else 0.50 if tier == 1 else 0.24
        if small_pre > 0.45 and tier <= 1:
            frac -= 0.14
        if rejam_rate > 0.22 and tier <= 1:
            frac -= 0.16 if tier == 0 else 0.10
        if state.cost_to_call == 0 and tier == 0:
            frac = min(frac, 0.20)
        frac += self.rng.uniform(-0.05, 0.05)
        return int(self._clip(lo + int(span * self._clip(frac, 0.06, 0.97)), lo, hi))

    def _post_raise_amount(self, state: PokerState, eq: float, rel_price: float, wet: int) -> int:
        lo, hi = state.raise_bounds
        span = max(0, hi - lo)
        frac = self._clip(0.58 + 0.30 * (eq - 0.50) + 0.10 * max(0.0, rel_price - 0.4) - 0.04 * wet, 0.18, 0.96)
        frac += self.rng.uniform(-0.06, 0.06)
        return int(self._clip(lo + int(span * self._clip(frac, 0.05, 0.99)), lo, hi))

    def _board_wetness(self, board: list[str]) -> int:
        if not board:
            return 0
        ranks = sorted([RANK_TO_INT[c[0]] for c in board], reverse=True)
        suits = [c[1] for c in board]
        max_suit = max(suits.count("s"), suits.count("h"), suits.count("d"), suits.count("c"))
        conn = sum(1 for i in range(len(ranks) - 1) if abs(ranks[i] - ranks[i + 1]) <= 2)
        paired = len(set(ranks)) < len(ranks)
        w = 0
        if max_suit >= 3:
            w += 1
        if conn >= 2:
            w += 1
        if paired:
            w += 1
        return w

    def _predict_opp_bid(self, opp_stack: int, pot: int) -> int:
        if self.opp_bid_exact:
            ex = sum(self.opp_bid_exact) / float(len(self.opp_bid_exact))
        else:
            ex = 0.10 * min(opp_stack, pot)
        if self.opp_bid_lb:
            lb = sum(self.opp_bid_lb) / float(len(self.opp_bid_lb))
            pred = 0.42 * ex + 0.58 * lb
            if lb > 1.35 * ex:
                pred = 0.35 * pred + 0.65 * lb
        else:
            pred = ex
        return int(self._clip(pred, 0, opp_stack))

    def _is_micro_bidder(self, pred: int, pot: int) -> bool:
        if not self.opp_bid_exact:
            return pred <= max(10, int(0.08 * pot))
        tiny = sum(1 for x in self.opp_bid_exact if x <= 8)
        return tiny / float(len(self.opp_bid_exact)) > 0.45

    def _is_high_mid_anchor_bidder(self, pred: int) -> bool:
        if len(self.opp_bid_exact) < 12:
            return False
        band = sum(1 for x in self.opp_bid_exact if 220 <= x <= 280)
        return band / float(len(self.opp_bid_exact)) > 0.30 and pred >= 170

    def _is_extreme_bidder(self, pred: int, pot: int) -> bool:
        if len(self.opp_bid_exact) < 6 and len(self.opp_bid_lb) < 6:
            return pred >= max(600, int(0.60 * pot))
        ref = list(self.opp_bid_exact) + list(self.opp_bid_lb)
        high = sum(1 for x in ref if x >= 500 or x >= int(0.45 * max(1, pot)))
        return high / float(max(1, len(ref))) > 0.35 and pred >= max(300, int(0.30 * pot))

    def _auction_loss_rate(self) -> float:
        if self.auction_total < 6:
            return 0.50
        return self.auction_opp_wins / float(max(1, self.auction_total))

    def _opp_high_bid_win_rate(self) -> float:
        if self.auction_total < 6:
            return 0.0
        return self.auction_high_opp_wins / float(max(1, self.auction_total))

    def _opp_fold_rate(self) -> float:
        if self.our_raise_opps < 8:
            return 0.34
        return self.opp_fold_to_raise_hits / float(max(1, self.our_raise_opps))

    def _mc_iters(self, time_bank: float, street: str) -> int:
        base = 44 if time_bank < 3.0 else 80 if time_bank < 7.0 else 130 if time_bank < 12.0 else 190
        if street == "river":
            return int(base * 0.70)
        if street == "turn":
            return int(base * 0.85)
        if street == "flop":
            return int(base * 1.08)
        return int(base * 0.92)

    def _estimate_equity(
        self,
        my_hand: list[str],
        board: list[str],
        opp_revealed: list[str],
        iters: int,
        t0: float,
        time_bank: float,
    ) -> float:
        key = (tuple(sorted(my_hand)), tuple(board), tuple(sorted(opp_revealed)), len(board), iters // 30)
        if key in self.equity_cache:
            return self.equity_cache[key]

        # Upload/runtime safety: if eval7 is unavailable in target environment,
        # fall back to a fast heuristic equity proxy instead of crashing.
        if eval7 is None:
            eq = self._heuristic_equity(my_hand, board, opp_revealed)
            self.equity_cache[key] = eq
            return eq

        known = my_hand + board + opp_revealed
        if len(set(known)) != len(known):
            return 0.5

        my_cards = [eval7.Card(c) for c in my_hand]
        board_cards = [eval7.Card(c) for c in board]
        rev_cards = [eval7.Card(c) for c in opp_revealed]
        all_cards = [eval7.Card(r + s) for r in RANK_ORDER for s in "shdc"]
        dead = set(my_cards + board_cards + rev_cards)
        rem = [c for c in all_cards if c not in dead]
        need_board = 5 - len(board_cards)

        budget = self._decision_budget(time_bank)
        wins = 0.0
        n = 0
        for _ in range(max(16, iters)):
            if time.perf_counter() - t0 > budget:
                break
            if len(rev_cards) == 2:
                opp = rev_cards
                board_draw = self.rng.sample(rem, need_board)
            elif len(rev_cards) == 1:
                drawn = self.rng.sample(rem, need_board + 1)
                opp = [rev_cards[0], drawn[0]]
                board_draw = drawn[1:]
            else:
                drawn = self.rng.sample(rem, need_board + 2)
                opp = [drawn[0], drawn[1]]
                board_draw = drawn[2:]

            full_board = board_cards + board_draw
            a = eval7.evaluate(my_cards + full_board)
            b = eval7.evaluate(opp + full_board)
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
            n += 1

        eq = wins / n if n > 0 else 0.5
        if len(self.equity_cache) > 7000:
            self.equity_cache.clear()
        self.equity_cache[key] = eq
        return eq

    def _heuristic_equity(self, my_hand: list[str], board: list[str], opp_revealed: list[str]) -> float:
        c1, c2 = my_hand
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        v1 = RANK_TO_INT[r1]
        v2 = RANK_TO_INT[r2]
        hi = max(v1, v2)
        lo = min(v1, v2)
        pair = v1 == v2
        suited = s1 == s2
        conn = abs(v1 - v2) <= 1

        eq = 0.50
        if pair:
            eq += 0.11 + 0.01 * max(0, hi - 8)
        else:
            eq += 0.015 * max(0, hi - 10)
            if suited:
                eq += 0.03
            if conn:
                eq += 0.02

        # Simple board interaction proxy.
        if board:
            board_ranks = [RANK_TO_INT[c[0]] for c in board]
            board_suits = [c[1] for c in board]
            if r1 in [c[0] for c in board] or r2 in [c[0] for c in board]:
                eq += 0.07
            if suited and board_suits.count(s1) >= 2:
                eq += 0.04
            if max(board_ranks) - min(board_ranks) <= 4:
                eq += 0.02

        # If one opponent card is revealed, discount when it dominates our high cards.
        if opp_revealed:
            ov = RANK_TO_INT[opp_revealed[0][0]]
            if ov >= hi and hi >= 11:
                eq -= 0.05
            elif hi > ov:
                eq += 0.02

        return float(self._clip(eq, 0.18, 0.85))

    @staticmethod
    def _decision_budget(time_bank: float) -> float:
        if time_bank < 1.8:
            return 0.010
        if time_bank < 5.0:
            return 0.018
        if time_bank < 10.0:
            return 0.030
        return 0.045

    # Small fixed-weight MLP-like scorer (neural-inspired).
    def _neural_score(self, eq_proxy: float, pot_odds: float, fold_rate: float, pressure: float, wetness: float) -> float:
        x = [eq_proxy, pot_odds, fold_rate, pressure, wetness, 1.0]
        h1 = math.tanh(1.4 * x[0] - 0.9 * x[1] + 0.7 * x[2] - 1.1 * x[3] - 0.3 * x[4] + 0.15)
        h2 = math.tanh(1.0 * x[0] + 0.6 * x[1] + 0.4 * x[2] - 0.7 * x[3] - 0.2 * x[4] - 0.10)
        h3 = math.tanh(0.8 * x[0] - 0.4 * x[1] + 0.9 * x[2] - 0.6 * x[3] - 0.1 * x[4] + 0.05)
        z = 1.2 * h1 + 0.8 * h2 + 0.9 * h3 - 0.15
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _rate(num: int, den: int, default: float) -> float:
        if den <= 0:
            return default
        return num / float(den)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x


if __name__ == "__main__":
    run_bot(Player(), parse_args())
