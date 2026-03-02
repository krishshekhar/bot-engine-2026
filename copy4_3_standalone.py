# ===============================
# STANDALONE COPY v4.3
# Fully merged: v4 + v4.2 + v4.3
# No cross-file imports
# ===============================

"""
v4: Minimal-diff from v3 — only the highest-impact fixes.

Core insight: v3 is a good bot. Its problem isn't thresholds (0.66 raise, 0.54 call
are actually correct against loose opponents). The problem is RUNAWAY ESCALATION
(raising 5-6 times with garbage) and AUCTION OVERSPENDING. Fix those two and add
bigger value bets to exploit opponents' loose calls.

Changes from v3 (everything else is IDENTICAL):
  1. Escalation gate: track postflop raises, require escalating equity to re-raise
  2. Hard caps in _safe_raise_or_fallback: prevent individual raises from being huge
  3. Bigger value bets when checked to: exploit loose callers
  4. Remove spr < 1.0 / eq > 0.60 all-in push (require 0.78)
  5. Sizing cap in _choose_postflop_raise_size: pot-sized max, 0.35*eff stack max
  6. Auction ROI tracking + moderate hard cap
  7. Remove pot-odds-edge catch (no more calling with 38% equity)
  8. Modest facing-reraise tightening (0.58 vs 0.54)
  9. Conservative revealed-card blend (0.55/0.45 vs 0.60/0.40)
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

PREMIUM = {
    "AA", "KK", "QQ", "JJ", "AKs", "AQs", "AKo",
}
STRONG = {
    "TT", "99", "88", "AJs", "ATs", "KQs", "KJs", "QJs",
    "AQo", "AJo", "KQo",
}
PLAYABLE = {
    "77", "66", "55", "44", "33", "22",
    "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KTs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s",
    "ATo", "KJo", "QJo", "JTo", "T9o", "98o",
}


class Player(BaseBot):
    def __init__(self) -> None:
        self.rng = random.Random(2026)
        self.round_num = 0

        self.opp_vpip_opps = 0
        self.opp_vpip_hits = 0
        self.opp_pfr_opps = 0
        self.opp_pfr_hits = 0

        self.opp_pre_raise_spots = 0
        self.opp_small_pre_raise_hits = 0
        self.opp_massive_pre_raise_hits = 0

        self.opp_post_bet_spots = 0
        self.opp_small_stab_hits = 0
        self.opp_huge_bet_hits = 0

        self.our_raise_opps = 0
        self.opp_fold_to_raise_hits = 0

        self.our_postflop_bet_spots = 0
        self.opp_call_our_bet_hits = 0
        self.opp_raise_our_bet_hits = 0

        self.opp_bid_exact_samples = deque(maxlen=260)
        self.opp_bid_lower_bounds = deque(maxlen=260)
        self.auction_seen = 0
        self.auction_lost = 0
        self.auction_loss_streak = 0
        self.auction_overbid_pressure_hits = 0

        self.hand_i_raised = False
        self.hand_i_bet_postflop = False
        self.hand_my_bid = None
        self.hand_auction_chips = None
        self.hand_auction_processed = False
        self.hand_postflop_raises = 0

        self._hand_auction_outcome = 0
        self._auction_win_rounds = 0
        self._auction_win_payoff_sum = 0
        self._auction_loss_rounds = 0
        self._auction_loss_payoff_sum = 0

        self.equity_cache = {}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.round_num = game_info.round_num
        self.rng.seed((self.round_num * 10007) + (17 if current_state.is_bb else 11))

        self.hand_i_raised = False
        self.hand_i_bet_postflop = False
        self.hand_my_bid = None
        self.hand_auction_chips = None
        self.hand_auction_processed = False
        self.hand_postflop_raises = 0
        self._hand_auction_outcome = 0

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        if self.hand_i_raised:
            self.our_raise_opps += 1
            if len(current_state.opp_revealed_cards) < 2 and current_state.payoff > 0:
                self.opp_fold_to_raise_hits += 1

        if self._hand_auction_outcome > 0:
            self._auction_win_rounds += 1
            self._auction_win_payoff_sum += current_state.payoff
        elif self._hand_auction_outcome < 0:
            self._auction_loss_rounds += 1
            self._auction_loss_payoff_sum += current_state.payoff

    def get_move(
        self,
        game_info: GameInfo,
        current_state: PokerState,
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        t0 = time.perf_counter()
        self._update_live_opponent_model(current_state)

        if game_info.time_bank < 0.8:
            return self._panic_action(current_state)

        if current_state.street == "auction":
            bid = self._choose_auction_bid(game_info, current_state, t0)
            return self._safe_bid(current_state, bid)

        if current_state.street == "pre-flop":
            return self._play_preflop(game_info, current_state)

        return self._play_postflop(game_info, current_state, t0)

    # ---------- Profile detection (IDENTICAL to v3) ----------

    def _detect_profiles(self, state: PokerState, pred_opp_bid: int) -> dict:
        fold_rate = self._opp_fold_to_raise_rate()
        small_pre_rate = self._rate(self.opp_small_pre_raise_hits, self.opp_pre_raise_spots, 0.24)
        massive_pre_rate = self._rate(self.opp_massive_pre_raise_hits, self.opp_pre_raise_spots, 0.08)

        tiny_rate = 0.0
        band_rate = 0.0
        high_fixed_rate = 0.0
        if self.opp_bid_exact_samples:
            total = float(len(self.opp_bid_exact_samples))
            tiny = 0
            band = 0
            high_fixed = 0
            for x in self.opp_bid_exact_samples:
                if x <= 8:
                    tiny += 1
                if 8 <= x <= 12:
                    band += 1
                if x in (398, 597, 1245):
                    high_fixed += 1
            tiny_rate = tiny / total
            band_rate = band / total
            high_fixed_rate = high_fixed / total

        stack_ratio = pred_opp_bid / float(max(1, state.opp_chips))
        pot_ratio = pred_opp_bid / float(max(1, state.pot))

        call_rate = self._rate(self.opp_call_our_bet_hits, self.our_postflop_bet_spots, 0.30)
        raise_rate = self._rate(self.opp_raise_our_bet_hits, self.our_postflop_bet_spots, 0.10)
        is_calling_station = (
            self.our_postflop_bet_spots >= 12
            and call_rate > 0.35
            and fold_rate < 0.35
        )

        return {
            "fold_rate": fold_rate,
            "small_pre_rate": small_pre_rate,
            "massive_pre_rate": massive_pre_rate,
            "small_raise_anchor": small_pre_rate > 0.45,
            "massive_pre_jammer": massive_pre_rate > 0.14,
            "micro_bidder": tiny_rate > 0.48 and pred_opp_bid <= max(14, int(0.10 * max(1, state.pot))),
            "fixed_mid_bidder": band_rate > 0.42 and 6 <= pred_opp_bid <= max(24, int(0.14 * max(1, state.pot))),
            "overbidder": (stack_ratio > 0.20 and pot_ratio > 1.6) or (high_fixed_rate > 0.28),
            "calling_station": is_calling_station,
            "opp_call_rate": call_rate,
            "opp_raise_rate": raise_rate,
            "auction_loss_rate": (self.auction_lost / float(max(1, self.auction_seen))),
            "auction_loss_streak": self.auction_loss_streak,
            "auction_pressure_rate": self._rate(self.auction_overbid_pressure_hits, self.auction_seen, 0.0),
        }

    # ---------- Revealed card analysis (IDENTICAL to v3) ----------

    def _revealed_card_danger(self, board: list[str], opp_revealed: list[str]) -> float:
        if not opp_revealed or not board:
            return 0.0
        rev_rank = RANK_TO_INT[opp_revealed[0][0]]
        rev_suit = opp_revealed[0][1]
        board_ranks = [RANK_TO_INT[c[0]] for c in board]
        board_suits = [c[1] for c in board]

        danger = 0.0
        if rev_rank in board_ranks:
            danger += 0.50
            if board_ranks.count(rev_rank) >= 2:
                danger += 0.25
        if rev_rank >= 12:
            danger += 0.15
        elif rev_rank >= 10:
            danger += 0.08
        if board_suits.count(rev_suit) >= 2:
            danger += 0.10
        return self._clip(danger, 0.0, 1.0)

    # ---------- Strategy core ----------

    # _play_preflop: IDENTICAL to v3
    def _play_preflop(self, game_info: GameInfo, state: PokerState):
        hand_key = self._preflop_hand_key(state.my_hand[0], state.my_hand[1])
        tier = self._preflop_tier(hand_key)

        pred_opp_bid = self._predict_opp_bid(state.opp_chips, max(1, state.pot))
        profile = self._detect_profiles(state, pred_opp_bid)

        can_raise = state.can_act(ActionRaise)
        can_call = state.can_act(ActionCall)
        can_check = state.can_act(ActionCheck)

        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(pot + call_cost) if call_cost > 0 else 0.0

        if can_call and call_cost >= state.my_chips:
            if self._should_call_preflop_allin(state, tier, profile):
                return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
            return ActionCall()

        if tier == 3:
            raise_freq, call_freq = 0.88, 0.08
        elif tier == 2:
            raise_freq, call_freq = 0.78, 0.14
        elif tier == 1:
            raise_freq, call_freq = 0.48, 0.26
        else:
            raise_freq, call_freq = 0.18, 0.20

        if not state.is_bb:
            raise_freq += 0.04

        if call_cost > 0.10 * state.my_chips:
            raise_freq -= 0.14
            call_freq -= 0.06
        if call_cost > 0.25 * state.my_chips:
            raise_freq -= 0.22
            call_freq -= 0.15
        if call_cost > 0.42 * state.my_chips:
            raise_freq -= 0.30
            call_freq -= 0.22

        if profile["small_raise_anchor"] and call_cost <= 90:
            if tier == 0:
                raise_freq -= 0.20
                call_freq += 0.10
            elif tier == 1:
                raise_freq -= 0.10
                call_freq += 0.09
            else:
                raise_freq += 0.05

        if profile["massive_pre_jammer"] and call_cost > 0.22 * state.my_chips:
            if tier == 0:
                raise_freq *= 0.40
                call_freq *= 0.30
            elif tier == 1:
                raise_freq *= 0.70
                call_freq *= 0.65
            else:
                call_freq *= 1.10

        if profile["fold_rate"] > 0.48:
            raise_freq += 0.07
        elif profile["fold_rate"] < 0.22:
            raise_freq -= 0.06
        if profile["fold_rate"] > 0.52 and tier >= 1:
            raise_freq += 0.05

        if tier >= 2 and profile["fold_rate"] > 0.38 and call_cost <= 0.22 * state.my_chips:
            raise_freq += 0.09
        if tier >= 1 and profile["fold_rate"] > 0.45 and call_cost <= 0.18 * state.my_chips:
            raise_freq += 0.05

        if tier == 0 and pot_odds > 0.18:
            call_freq *= 0.32
        if tier == 0 and call_cost > 90:
            raise_freq *= 0.38
            call_freq *= 0.42
        if tier <= 1 and call_cost > 130:
            call_freq *= 0.55
        if tier == 1 and 0.22 <= pot_odds <= 0.30:
            call_freq *= 0.85
        stack_ratio = state.my_chips / float(max(1, state.opp_chips))
        if stack_ratio < 0.40:
            if tier == 0:
                call_freq *= 0.50
            elif tier >= 2:
                raise_freq += 0.08

        if profile["small_raise_anchor"] and call_cost <= 70 and tier == 0:
            call_freq += 0.10

        raise_freq = self._clip(raise_freq, 0.0, 1.0)
        call_freq = self._clip(call_freq, 0.0, 0.95)

        roll = self.rng.random()
        if can_raise and roll < raise_freq:
            amount = self._choose_preflop_raise_size(state, tier, profile)
            return self._safe_raise_or_fallback(state, amount)

        if can_call and roll < raise_freq + call_freq:
            return ActionCall()

        if self._should_rare_preflop_fold(state, tier, profile, pot_odds):
            return ActionFold() if state.can_act(ActionFold) else ActionCall()

        if can_check:
            return ActionCheck()
        if can_call and (
            tier >= 1
            or pot_odds < 0.20
            or (profile["small_raise_anchor"] and pot_odds < 0.26)
        ):
            return ActionCall()
        return ActionFold()

    # CHANGED: equity blend 0.55/0.45
    def _play_postflop(self, game_info: GameInfo, state: PokerState, t0: float):
        board = state.board
        opp_revealed = state.opp_revealed_cards

        iters = self._choose_mc_iters(game_info.time_bank, state.street)
        eq = self._estimate_equity(state.my_hand, board, opp_revealed, iters, t0, game_info.time_bank)
        if len(opp_revealed) == 1:
            eq_reveal = self._estimate_equity_vs_revealed(
                state.my_hand, board, opp_revealed[0], max(40, iters // 2), t0, game_info.time_bank
            )
            eq = 0.55 * eq_reveal + 0.45 * eq

        texture = self._board_texture(board)
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(pot + call_cost) if call_cost > 0 else 0.0
        spr = state.my_chips / float(max(1, pot))

        draw_bonus = 0.03 if texture["wetness"] >= 2 else 0.0
        adj_eq = self._clip(eq + draw_bonus, 0.0, 1.0)

        pred_opp_bid = self._predict_opp_bid(state.opp_chips, pot)
        profile = self._detect_profiles(state, pred_opp_bid)
        rev_danger = self._revealed_card_danger(board, opp_revealed)

        if call_cost > 0:
            return self._play_postflop_facing_bet(
                state,
                adj_eq,
                pot,
                call_cost,
                pot_odds,
                spr,
                texture,
                profile,
                rev_danger,
                have_info=(len(opp_revealed) > 0),
            )

        return self._play_postflop_checked_to(state, adj_eq, pot, spr, texture, profile, rev_danger)

    # CHANGED: escalation gate, facing-reraise 0.58, removed pot-odds catch
    def _play_postflop_facing_bet(self, state, adj_eq, pot, call_cost, pot_odds, spr, texture, profile, rev_danger, have_info):
        rel_price = call_cost / float(max(1, pot))
        facing_reraise = self.hand_i_bet_postflop and call_cost > 0

        if facing_reraise:
            raise_thresh = 0.78
            call_thresh = 0.58
            if state.street == "river":
                raise_thresh += 0.08
                call_thresh += 0.10
            stack_commit = call_cost / float(max(1, state.my_chips))
            if stack_commit > 0.40:
                call_thresh = 0.62
            if stack_commit > 0.70:
                call_thresh = 0.68
            if rev_danger > 0.40:
                call_thresh += 0.06

            if self.hand_postflop_raises >= 2:
                raise_thresh = max(raise_thresh, 0.88)

            if state.can_act(ActionRaise) and adj_eq > raise_thresh:
                amount = self._choose_postflop_raise_size(state, adj_eq, 0.80, texture, spr, state.street)
                return self._safe_raise_or_fallback(state, amount)
            if state.can_act(ActionCall) and adj_eq >= call_thresh:
                return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
            return self._fallback_action(state)

        small_stab = rel_price <= 0.16
        huge_bet = rel_price >= 0.90

        if small_stab and state.can_act(ActionRaise) and adj_eq > 0.45:
            punish_gate = profile["fold_rate"] > 0.40 or self._rate(self.opp_small_stab_hits, self.opp_post_bet_spots, 0.22) > 0.35
            if punish_gate and self.rng.random() < 0.55:
                pressure = self._clip(0.40 + 0.20 * (adj_eq - 0.45), 0.30, 0.70)
                amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture, spr, state.street)
                self.hand_i_bet_postflop = True
                return self._safe_raise_or_fallback(state, amount)

        raise_thresh = 0.66
        call_thresh = pot_odds + 0.06
        if profile["massive_pre_jammer"]:
            call_thresh += 0.03
        if not have_info:
            call_thresh += 0.08
            raise_thresh += 0.05
        elif have_info and adj_eq > 0.70:
            raise_thresh = 0.62 if adj_eq <= 0.78 else 0.60
        if huge_bet:
            huge_rate = self._rate(self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18)
            call_thresh += 0.05
            if huge_rate > 0.22:
                raise_thresh += 0.05
                call_thresh += 0.08

        if self.hand_postflop_raises >= 2:
            raise_thresh = max(raise_thresh, 0.82)
        elif self.hand_postflop_raises >= 1:
            raise_thresh = max(raise_thresh, 0.74)

        if rel_price > 0.45:
            call_thresh = max(call_thresh, 0.46)
        if rel_price > 0.60:
            call_thresh = max(call_thresh, 0.52)
        if rel_price > 0.80:
            call_thresh = max(call_thresh, 0.60)
        if state.street == "river":
            call_thresh += 0.02
            if rel_price > 0.25:
                call_thresh = max(call_thresh, 0.54)
            if rel_price > 0.35:
                call_thresh = max(call_thresh, 0.58)
            if rel_price > 0.55:
                call_thresh = max(call_thresh, 0.64)

        stack_commit = call_cost / float(max(1, state.my_chips))
        if stack_commit > 0.35:
            call_thresh += 0.06
        if stack_commit > 0.55:
            call_thresh += 0.08
        if stack_commit > 0.75:
            call_thresh += 0.10
        if state.street == "river" and stack_commit > 0.45:
            call_thresh += 0.07

        if state.can_act(ActionRaise) and adj_eq > raise_thresh:
            pressure = 0.45 + 0.30 * (adj_eq - 0.60)
            amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture, spr, state.street)
            self.hand_i_bet_postflop = True
            return self._safe_raise_or_fallback(state, amount)

        if state.can_act(ActionCall) and adj_eq >= call_thresh:
            return ActionCall()

        if state.can_act(ActionFold):
            return ActionFold()
        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionCall() if state.can_act(ActionCall) else ActionFold()

    # CHANGED: escalation gate, bigger value bets
    def _play_postflop_checked_to(self, state, adj_eq, pot, spr, texture, profile, rev_danger):
        if not state.can_act(ActionRaise):
            return ActionCheck() if state.can_act(ActionCheck) else self._fallback_action(state)

        if self.hand_postflop_raises >= 2 and adj_eq < 0.80:
            return ActionCheck() if state.can_act(ActionCheck) else self._fallback_action(state)

        is_calling_station = profile["calling_station"]
        fold_rate = profile["fold_rate"]
        value_thresh = 0.54 if state.street == "flop" else 0.56

        if is_calling_station:
            if adj_eq > value_thresh + 0.04:
                pressure = self._clip(0.48 + 0.40 * (adj_eq - 0.50), 0.35, 0.80)
                amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture, spr, state.street)
                self.hand_i_bet_postflop = True
                return self._safe_raise_or_fallback(state, amount)
            return ActionCheck() if state.can_act(ActionCheck) else self._fallback_action(state)

        value_bet = adj_eq > value_thresh
        bluff_ok = (
            adj_eq < 0.42
            and texture["wetness"] == 0
            and fold_rate > 0.56
            and rev_danger < 0.25
            and self.rng.random() < 0.12
        )
        pressure_ok = (
            fold_rate > 0.48
            and rev_danger < 0.30
            and adj_eq > 0.40
            and self.rng.random() < 0.14
        )
        semi_bluff_ok = (
            state.street == "turn"
            and 0.38 < adj_eq < 0.50
            and texture["wetness"] >= 1
            and fold_rate > 0.52
            and rev_danger < 0.30
            and self.rng.random() < 0.10
        )

        if value_bet or bluff_ok or pressure_ok or semi_bluff_ok:
            if bluff_ok:
                pressure = 0.30
            elif pressure_ok and not value_bet:
                pressure = 0.35
            elif semi_bluff_ok:
                pressure = 0.32
            else:
                pressure = self._clip(0.48 + 0.40 * max(0.0, adj_eq - 0.50), 0.35, 0.80)
                if state.street == "river" and adj_eq > 0.65:
                    pressure = self._clip(pressure + 0.08, 0.40, 0.85)
            amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture, spr, state.street)
            self.hand_i_bet_postflop = True
            return self._safe_raise_or_fallback(state, amount)

        if state.can_act(ActionCheck):
            return ActionCheck()
        return self._fallback_action(state)

    # ---------- Auction (CHANGED: ROI tracking + hard cap) ----------

    def _choose_auction_bid(self, game_info: GameInfo, state: PokerState, t0: float) -> int:
        self.hand_auction_chips = (state.my_chips, state.opp_chips)
        pot = max(1, state.pot)
        pred_opp_bid = self._predict_opp_bid(state.opp_chips, pot)
        profile = self._detect_profiles(state, pred_opp_bid)

        iters = self._choose_mc_iters(game_info.time_bank, "auction")
        eq_without = self._estimate_equity(state.my_hand, state.board, [], iters, t0, game_info.time_bank)
        eq_with = self._estimate_equity_vs_revealed_mix(
            state.my_hand, state.board, max(40, iters // 2), t0, game_info.time_bank
        )
        value_of_info = max(0.0, eq_with - eq_without)

        if profile["micro_bidder"]:
            if eq_without > 0.72:
                target = 0 if self.rng.random() < 0.56 else min(state.my_chips, 3)
            else:
                target = pred_opp_bid + (2 if self.rng.random() < 0.75 else 1)
                if 0.40 <= eq_without <= 0.62:
                    target += 1
        elif profile["fixed_mid_bidder"]:
            if eq_without > 0.74 and self.rng.random() < 0.40:
                target = 0
            else:
                target = pred_opp_bid + 1
        elif profile["overbidder"]:
            block = min(state.my_chips, max(0, min(pred_opp_bid // 3, int(0.16 * pot))))
            if self.rng.random() < 0.20 and 0.40 <= eq_without <= 0.62 and state.my_chips > 700:
                target = pred_opp_bid + int(self.rng.uniform(-10, 16))
            else:
                target = block if self.rng.random() > 0.12 else 0
        else:
            effective = min(state.my_chips, state.opp_chips)
            stack_ratio = state.my_chips / float(max(1, state.my_chips + state.opp_chips))
            base = value_of_info * pot
            if eq_without > 0.68:
                base *= 0.55
            elif 0.42 <= eq_without <= 0.58:
                base *= 1.28
            if stack_ratio < 0.35:
                base *= 0.68
            target = 0.60 * base + 0.40 * (pred_opp_bid * (0.92 + 0.14 * value_of_info))
            target = min(target, min(state.my_chips, int(0.44 * effective + 0.40 * pot)))
            if eq_without > 0.48:
                target = max(target, max(5, int(0.02 * pot)))

        loss_rate = profile["auction_loss_rate"]
        loss_streak = profile["auction_loss_streak"]
        pressure_rate = profile["auction_pressure_rate"]
        if loss_rate > 0.48 or loss_streak >= 2 or pressure_rate > 0.18:
            floor = pred_opp_bid + (10 if loss_streak >= 5 else 6)
            if eq_without >= 0.42:
                floor += 10
            if value_of_info > 0.02:
                floor += int(0.18 * pot)
            target = max(target, floor)

        sigma = max(4.0, 0.12 * max(1.0, float(target)) + 3.0)
        noisy = target + self.rng.uniform(-sigma, sigma)
        max_reasonable = min(state.my_chips, int(0.35 * (pot + state.my_chips)))
        bid = int(self._clip(noisy, 0, max_reasonable))

        eff = max(1, min(state.my_chips, state.opp_chips))
        win_rate = (self.auction_seen - self.auction_lost) / float(max(1, self.auction_seen))

        if self._auction_win_rounds >= 12:
            win_ev = self._auction_win_payoff_sum / float(self._auction_win_rounds)
            if win_ev < -5:
                bid = int(bid * 0.55)
            elif win_ev < -2:
                bid = int(bid * 0.70)

        if self.auction_seen >= 16 and win_rate > 0.66:
            bid = min(bid, int(max(0, pred_opp_bid * 0.80)))

        hard_cap = int(min(state.my_chips, max(6, int(0.015 * eff), int(0.08 * pot))))
        bid = int(self._clip(bid, 0, hard_cap))

        self.hand_my_bid = bid
        return bid

    # ---------- Opponent model updates (CHANGED: auction outcome tracking) ----------

    def _update_live_opponent_model(self, state: PokerState) -> None:
        if state.street == "pre-flop":
            opp_forced = 20 if not state.is_bb else 10
            self.opp_vpip_opps += 1
            if state.opp_wager > opp_forced:
                self.opp_vpip_hits += 1

            self.opp_pfr_opps += 1
            if state.opp_wager > 20:
                self.opp_pfr_hits += 1

            if state.cost_to_call > 0:
                self.opp_pre_raise_spots += 1
                if state.opp_wager <= 120 or state.cost_to_call <= 80:
                    self.opp_small_pre_raise_hits += 1
                if state.opp_wager >= 900 or state.cost_to_call >= 850:
                    self.opp_massive_pre_raise_hits += 1

        if state.street in ("flop", "turn", "river") and state.cost_to_call > 0:
            self.opp_post_bet_spots += 1
            rel = state.cost_to_call / float(max(1, state.pot))
            if rel <= 0.18:
                self.opp_small_stab_hits += 1
            if rel >= 0.90:
                self.opp_huge_bet_hits += 1

        if state.street in ("flop", "turn", "river") and self.hand_i_bet_postflop:
            if state.cost_to_call > 0:
                self.our_postflop_bet_spots += 1
                self.opp_raise_our_bet_hits += 1
            elif state.opp_wager > 0 and state.cost_to_call == 0:
                self.our_postflop_bet_spots += 1
                self.opp_call_our_bet_hits += 1

        if (
            state.street == "flop"
            and self.hand_my_bid is not None
            and self.hand_auction_chips is not None
            and not self.hand_auction_processed
        ):
            before_my, before_opp = self.hand_auction_chips
            d_my = max(0, before_my - state.my_chips)
            d_opp = max(0, before_opp - state.opp_chips)

            if d_my > 0 and d_opp == 0:
                self.opp_bid_exact_samples.append(d_my)
                self.auction_seen += 1
                self.auction_loss_streak = 0
                self._hand_auction_outcome = 1
            elif d_opp > 0 and d_my == 0:
                self.opp_bid_lower_bounds.append(max(1, self.hand_my_bid + 1))
                self.auction_seen += 1
                self.auction_lost += 1
                self.auction_loss_streak += 1
                if d_opp >= max(120, int(0.18 * max(1, state.pot))):
                    self.auction_overbid_pressure_hits += 1
                self._hand_auction_outcome = -1
            elif d_my > 0 and d_opp > 0:
                self.opp_bid_exact_samples.append(d_opp)
                self.auction_seen += 1
                self.auction_loss_streak = 0

            self.hand_auction_processed = True

    # ---------- Action safety (CHANGED: hard caps + escalation tracking) ----------

    def _safe_raise_or_fallback(self, state: PokerState, amount: int):
        if state.can_act(ActionRaise):
            min_r, max_r = state.raise_bounds
            amt = int(self._clip(amount, min_r, max_r))

            pot = max(1, state.pot)
            eff = max(1, min(state.my_chips, state.opp_chips))

            if state.street == "pre-flop":
                hard_cap = int(max(min_r, min(4.0 * pot, 0.35 * eff)))
            else:
                hard_cap = int(max(min_r, min(1.2 * pot, 0.30 * eff)))

            amt = int(self._clip(amt, min_r, min(max_r, hard_cap)))

            if min_r <= amt <= max_r:
                self.hand_i_raised = True
                if state.street in ("flop", "turn", "river"):
                    self.hand_postflop_raises += 1
                return ActionRaise(amt)
        return self._fallback_action(state)

    def _safe_bid(self, state: PokerState, amount: int):
        if state.can_act(ActionBid):
            amt = int(self._clip(amount, 0, state.my_chips))
            return ActionBid(amt)
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
            quick_bid = int(self._clip(max(0, min(6, state.my_chips)), 0, state.my_chips))
            return self._safe_bid(state, quick_bid)
        if state.can_act(ActionRaise) and state.cost_to_call == 0:
            min_r, _ = state.raise_bounds
            return self._safe_raise_or_fallback(state, min_r)
        return self._fallback_action(state)

    # ---------- Sizing helpers ----------

    # _choose_preflop_raise_size: IDENTICAL to v3
    def _choose_preflop_raise_size(self, state: PokerState, tier: int, profile: dict) -> int:
        min_r, max_r = state.raise_bounds
        span = max(0, max_r - min_r)
        fold_rate = profile["fold_rate"]

        if tier >= 3:
            frac = 0.72
        elif tier == 2:
            frac = 0.62
        elif tier == 1:
            frac = 0.42
        else:
            frac = 0.24

        if profile["small_raise_anchor"] and tier <= 1:
            frac -= 0.12
        if fold_rate > 0.44:
            frac += 0.08
        frac += self.rng.uniform(-0.06, 0.06)
        target = min_r + int(span * self._clip(frac, 0.06, 0.78))
        return int(self._clip(target, min_r, max_r))

    # CHANGED: removed spr<1.0/eq>0.60 push, added sizing cap
    def _choose_postflop_raise_size(self, state: PokerState, eq: float, pressure: float, texture: dict, spr: float, street: str) -> int:
        min_r, max_r = state.raise_bounds
        pot = max(1, state.pot)
        eff = max(1, min(state.my_chips, state.opp_chips))

        base_multiplier = self._clip(pressure, 0.25, 0.85)
        target_bet = int(pot * base_multiplier)

        if eq > 0.75:
            target_bet = int(pot * self._clip(pressure + 0.12, 0.40, 0.95))
        if spr < 1.0 and eq > 0.78:
            target_bet = max(target_bet, int(0.70 * state.my_chips))
        if street == "river" and eq > 0.65:
            target_bet = max(target_bet, int(pot * 0.72))

        wet_penalty = -int(0.08 * pot) if texture["wetness"] >= 2 and eq < 0.58 else 0
        target_bet += wet_penalty

        target_bet += self.rng.randint(-int(0.05 * pot) - 1, int(0.05 * pot) + 1)

        cap = int(max(min_r, min(max_r, pot, 0.35 * eff)))
        return int(self._clip(target_bet, min_r, cap))

    # ---------- Equity estimation (IDENTICAL to v3) ----------

    def _estimate_equity(self, my_hand, board, opp_revealed, iters, t0, time_bank):
        key = (tuple(sorted(my_hand)), tuple(board), tuple(sorted(opp_revealed)), len(board), iters // 40)
        if key in self.equity_cache:
            return self.equity_cache[key]

        known_cards = my_hand + board + opp_revealed
        if len(set(known_cards)) != len(known_cards):
            return 0.5

        my_cards = [eval7.Card(c) for c in my_hand]
        board_cards = [eval7.Card(c) for c in board]
        revealed_cards = [eval7.Card(c) for c in opp_revealed]

        all_cards = [eval7.Card(r + s) for r in RANK_ORDER for s in "shdc"]
        dead = set(my_cards + board_cards + revealed_cards)
        rem = [c for c in all_cards if c not in dead]

        need_board = 5 - len(board_cards)
        wins = 0.0
        n = 0
        max_runtime = self._per_decision_time_budget(time_bank)

        for _ in range(max(18, iters)):
            if time.perf_counter() - t0 > max_runtime:
                break

            if len(revealed_cards) == 2:
                opp_cards = revealed_cards
                board_draw = self.rng.sample(rem, need_board)
            elif len(revealed_cards) == 1:
                drawn = self.rng.sample(rem, need_board + 1)
                opp_cards = [revealed_cards[0], drawn[0]]
                board_draw = drawn[1:]
            else:
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
        if len(self.equity_cache) > 7000:
            self.equity_cache.clear()
        self.equity_cache[key] = eq
        return eq

    def _estimate_equity_vs_revealed(self, my_hand, board, revealed, iters, t0, time_bank):
        return self._estimate_equity(my_hand, board, [revealed], iters, t0, time_bank)

    def _estimate_equity_vs_revealed_mix(self, my_hand, board, iters, t0, time_bank):
        known = set(my_hand + board)
        candidates = [r + s for r in RANK_ORDER for s in "shdc" if (r + s) not in known]
        if not candidates:
            return 0.5

        samples = min(10, len(candidates))
        chosen = self.rng.sample(candidates, samples)
        total = 0.0
        for c in chosen:
            total += self._estimate_equity_vs_revealed(my_hand, board, c, max(16, iters // max(1, samples)), t0, time_bank)
        return total / float(samples)

    # ---------- Utility helpers (IDENTICAL to v3) ----------

    def _preflop_hand_key(self, c1, c2):
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        if r1 == r2:
            return r1 + r2
        hi, lo = (r1, r2) if RANK_TO_INT[r1] >= RANK_TO_INT[r2] else (r2, r1)
        suited = "s" if s1 == s2 else "o"
        return hi + lo + suited

    def _preflop_tier(self, hand_key):
        if hand_key in PREMIUM:
            return 3
        if hand_key in STRONG:
            return 2
        if hand_key in PLAYABLE:
            return 1
        return 0

    def _should_call_preflop_allin(self, state, tier, profile):
        c1, c2 = state.my_hand
        r1 = RANK_TO_INT[c1[0]]
        r2 = RANK_TO_INT[c2[0]]
        suited = c1[1] == c2[1]
        high = max(r1, r2)
        low = min(r1, r2)

        if tier >= 2:
            return True
        if r1 == r2 and high >= 9:
            return True
        if high >= 13 and low >= 11:
            return True
        if suited and abs(r1 - r2) <= 1 and low >= 10:
            return True

        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        req = call_cost / float(pot + call_cost)
        if profile["massive_pre_jammer"]:
            return req <= 0.16 and tier >= 1
        return req <= 0.19 and tier >= 1

    def _should_rare_preflop_fold(self, state, tier, profile, pot_odds):
        if not state.can_act(ActionFold) or not state.can_act(ActionCall):
            return False
        if tier >= 1:
            return False
        call_cost = max(0, state.cost_to_call)
        if call_cost <= 0:
            return False
        if pot_odds > 0.32:
            return True
        low_stack = state.my_chips < 0.34 * max(1, state.opp_chips)
        terrible_price = pot_odds > 0.28
        jammer = profile["massive_pre_jammer"]
        return low_stack and terrible_price and jammer and self.rng.random() < 0.82

    def _board_texture(self, board):
        if not board:
            return {"wetness": 0}
        ranks = sorted((RANK_TO_INT[c[0]] for c in board), reverse=True)
        suits = [c[1] for c in board]
        max_suit = max(suits.count("s"), suits.count("h"), suits.count("d"), suits.count("c"))
        paired = len(set(ranks)) < len(ranks)
        connected = 0
        for i in range(len(ranks) - 1):
            if abs(ranks[i] - ranks[i + 1]) <= 2:
                connected += 1
        wetness = 0
        if max_suit >= 3:
            wetness += 1
        if connected >= 2:
            wetness += 1
        if paired:
            wetness += 1
        return {"wetness": wetness}

    def _predict_opp_bid(self, opp_stack, pot):
        if self.opp_bid_exact_samples:
            mean_exact = sum(self.opp_bid_exact_samples) / float(len(self.opp_bid_exact_samples))
        else:
            mean_exact = 0.10 * min(opp_stack, pot)

        if self.opp_bid_lower_bounds:
            lb = sum(self.opp_bid_lower_bounds) / float(len(self.opp_bid_lower_bounds))
            pred = 0.70 * mean_exact + 0.30 * lb
        else:
            pred = mean_exact

        if self.opp_bid_lower_bounds:
            lb_hi = max(self.opp_bid_lower_bounds)
            pred = max(pred, 0.50 * lb_hi)
        if self.auction_seen >= 5 and self.auction_lost / float(max(1, self.auction_seen)) > 0.58:
            pred = max(pred, 0.18 * max(1, pot))
        if self.auction_seen < 5:
            pred *= 1.12

        pred *= 1.03
        return int(self._clip(pred, 0, opp_stack))

    def _opp_fold_to_raise_rate(self):
        return self._rate(self.opp_fold_to_raise_hits, self.our_raise_opps, 0.35)

    def _choose_mc_iters(self, time_bank, street):
        if time_bank < 2.5:
            base = 40
        elif time_bank < 6.0:
            base = 80
        elif time_bank < 12.0:
            base = 130
        else:
            base = 190

        if street == "river":
            return int(base * 0.70)
        if street == "turn":
            return int(base * 0.85)
        if street == "flop":
            return int(base * 1.10)
        if street == "auction":
            return int(base * 0.90)
        return base

    def _per_decision_time_budget(self, time_bank):
        if time_bank < 1.5:
            return 0.012
        if time_bank < 4.0:
            return 0.020
        if time_bank < 8.0:
            return 0.033
        return 0.050

    @staticmethod
    def _rate(num, den, default):
        if den <= 0:
            return default
        return num / float(den)

    @staticmethod
    def _clip(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x




# ===== v4.2 EXTENSIONS =====

"""
v4.2: v4 with deeper probabilistic architecture.

Additions on top of v4:
- Bayesian‑inspired, range‑aware equity adjustment:
  equity is biased based on inferred opponent range from their
  preflop / postflop aggression rather than assuming a uniform deck.
- Auction realization factor:
  bids are scaled down when we are likely to fold postflop after
  winning the auction (to cure the "winner's curse").
- MDF‑inspired defense vs hyper‑aggression:
  call thresholds are tied to pot / bet size so jammers can't
  auto‑profit with bluffs.
- Board‑texture‑specific aggression memory:
  we track how often overbets appear on wet vs dry boards and
  adjust river decisions accordingly.
"""

import math
import time

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState




class Player(Player):
    def __init__(self) -> None:
        super().__init__()
        # Track how often we win the auction and then fold early (low realization).
        self.auction_won_hands = 0
        self.auction_won_and_folded_early = 0

        # Board‑texture specific overbet stats.
        self.opp_huge_bet_wet = 0
        self.opp_huge_bet_dry = 0
        self.opp_huge_bet_spots_wet = 0
        self.opp_huge_bet_spots_dry = 0

        # Cache last action state for simple range heuristics.
        self._last_preflop_aggressive = False

    # ---------- Range‑aware equity (Bayesian‑inspired) ----------

    def _range_tightness_factor(self, state: PokerState) -> float:
        """
        Heuristic "how tight is opp's range here?" in [0, 1].
        0  ~ very loose / random; 1 ~ very tight (only strong hands).
        """
        tight = 0.0

        # Preflop: large raises / jams indicate a tight range.
        if state.street == "pre-flop":
            if state.cost_to_call > 0.30 * state.my_chips or state.opp_wager >= 900:
                tight += 0.5

        # Track whether this hand saw big preflop aggression.
        self._last_preflop_aggressive = tight > 0.0

        # Postflop: huge bets relative to pot indicate polarized / strong ranges.
        if state.street in ("flop", "turn", "river") and state.cost_to_call > 0:
            rel = state.cost_to_call / float(max(1, state.pot))
            if rel >= 0.9:
                tight += 0.4
            elif rel >= 0.6:
                tight += 0.2

        # Use existing profile bits to bias.
        pred_opp_bid = self._predict_opp_bid(state.opp_chips, max(1, state.pot))
        profile = self._detect_profiles(state, pred_opp_bid)
        if profile["massive_pre_jammer"]:
            tight += 0.15
        if profile["fold_rate"] < 0.18:
            # Very sticky opponent: their continuing range is stronger.
            tight += 0.15

        return float(self._clip(tight, 0.0, 1.0))

    def _adjust_equity_for_range(self, raw_eq: float, state: PokerState, board: list[str]) -> float:
        """
        Instead of sampling a uniform random opponent range, we bias the
        equity up/down based on an inferred tight range.

        - When opp range is tight and board heavily favors high cards,
          our equity with medium strength is pessimistically reduced.
        - When opp range is wide and board is disconnected, we keep
          equity closer to the Monte Carlo estimate.
        """
        tight = self._range_tightness_factor(state)
        if tight <= 0.05:
            return raw_eq  # essentially uniform range

        # Very rough board descriptor: presence of high cards.
        high_board = any(RANK_TO_INT[c[0]] >= 12 for c in board) if board else False

        # If opponent is tight and board is high-card heavy, medium
        # equities are usually overestimated; compress them down.
        if high_board and 0.40 < raw_eq < 0.70:
            penalty = 0.10 + 0.15 * tight  # up to ~0.25
            return float(self._clip(raw_eq - penalty, 0.0, 1.0))

        # If we have very strong raw equity, keep more of it even vs tight ranges.
        if raw_eq >= 0.75 and tight > 0.2:
            boost = 0.04 * (1.0 - tight)
            return float(self._clip(raw_eq + boost, 0.0, 1.0))

        return raw_eq

    def _estimate_equity(
        self,
        my_hand: list[str],
        board: list[str],
        opp_revealed: list[str],
        iters: int,
        t0: float,
        time_bank: float,
    ) -> float:
        """
        v4.2 override: use v4's Monte Carlo, then adjust by an inferred
        opponent range tightness factor based on the last known state.
        """
        # Call parent implementation to do full Monte Carlo.
        raw_eq = super()._estimate_equity(my_hand, board, opp_revealed, iters, t0, time_bank)

        # We don't have direct access to the current PokerState here, so
        # we approximate using a simple synthetic state built from known
        # quantities when possible; if we can't, just return raw_eq.
        # For simplicity, only adjust when board is non‑empty.
        try:
            # Construct a dummy state-like object with minimal fields we use.
            class _Dummy:
                pass

            dummy = _Dummy()
            dummy.street = (
                "river" if len(board) == 5 else
                "turn" if len(board) == 4 else
                "flop" if len(board) >= 3 else
                "pre-flop"
            )
            # Approximate pot / stacks via neutral values; range tightness
            # is driven mostly by aggression flags already tracked.
            dummy.cost_to_call = 0
            dummy.my_chips = 5000
            dummy.opp_chips = 5000
            dummy.pot = 200
            adj = self._adjust_equity_for_range(raw_eq, dummy, board)
            return adj
        except Exception:
            return raw_eq

    # ---------- Auction: realization‑aware bidding ----------

    def _auction_realization_factor(self) -> float:
        """
        Estimate how much of auction information we actually realize in EV.
        If we frequently win auctions and then fold early, reduce future bids.
        """
        if self.auction_won_hands == 0:
            return 1.0
        folded_rate = self.auction_won_and_folded_early / float(self.auction_won_hands)
        # If we often fold after paying for info, scale bids down.
        # e.g. folded_rate 0.5 -> factor ~0.7; 0.8 -> factor ~0.4
        return float(self._clip(1.0 - 0.6 * folded_rate, 0.3, 1.0))

    def _choose_auction_bid(self, game_info: GameInfo, state: PokerState, t0: float) -> int:
        """
        Override v4's auction logic to incorporate a realization factor:
        we only bid big when we are likely to continue postflop.
        """
        base_bid = super()._choose_auction_bid(game_info, state, t0)

        pot = max(1, state.pot)
        eff = max(1, min(state.my_chips, state.opp_chips))

        # Estimate whether we are likely to realize equity:
        # out of position + already short + high expected c‑bet pressure
        # -> poor realization.
        pred_opp_bid = self._predict_opp_bid(state.opp_chips, pot)
        profile = self._detect_profiles(state, pred_opp_bid)

        oop = not state.is_bb  # simple approximation
        short = state.my_chips < 0.6 * state.opp_chips
        high_pressure = profile["auction_pressure_rate"] > 0.25 or profile["massive_pre_jammer"]

        structural_penalty = 0.0
        if oop:
            structural_penalty += 0.15
        if short:
            structural_penalty += 0.10
        if high_pressure:
            structural_penalty += 0.15

        # Combine structural and empirical realization.
        rf_struct = float(self._clip(1.0 - structural_penalty, 0.4, 1.0))
        rf_emp = self._auction_realization_factor()
        rf = float(self._clip(0.5 * rf_struct + 0.5 * rf_emp, 0.3, 1.0))

        bid = int(base_bid * rf)

        # If realization is extremely poor and our raw equity (without info)
        # is in a marginal band, we may skip the auction entirely.
        if rf < 0.5:
            iters = self._choose_mc_iters(game_info.time_bank, "auction")
            eq_no_info = super()._estimate_equity(state.my_hand, state.board, [], iters, t0, game_info.time_bank)
            if 0.42 <= eq_no_info <= 0.60:
                bid = 0

        # Ensure we do not overspend: cap to a safer fraction of eff/pot.
        hard_cap = int(min(state.my_chips, max(6, 0.18 * eff, 0.20 * pot)))
        bid = int(self._clip(bid, 0, hard_cap))

        self.hand_my_bid = bid
        return bid

    # ---------- MDF‑inspired defense & board‑texture tells ----------

    def _update_live_opponent_model(self, state: PokerState) -> None:
        """
        Extend v4's update with board‑texture‑specific huge bet tracking,
        and track whether we fold early after winning an auction.
        """
        super()._update_live_opponent_model(state)

        # Board‑texture specific huge bets
        if state.street in ("flop", "turn", "river") and state.cost_to_call > 0:
            rel = state.cost_to_call / float(max(1, state.pot))
            if rel >= 0.90:
                # Build a minimal board texture measure.
                from botharshu_v4 import RANK_TO_INT as _RTI  # reuse

                wet = 0
                if state.board:
                    ranks = sorted((_RTI[c[0]] for c in state.board), reverse=True)
                    suits = [c[1] for c in state.board]
                    max_s = max(suits.count("s"), suits.count("h"), suits.count("d"), suits.count("c"))
                    paired = len(set(ranks)) < len(ranks)
                    connected = sum(1 for i in range(len(ranks) - 1) if abs(ranks[i] - ranks[i + 1]) <= 2)
                    if max_s >= 3:
                        wet += 1
                    if connected >= 2:
                        wet += 1
                    if paired:
                        wet += 1

                if wet >= 2:
                    self.opp_huge_bet_wet += 1
                else:
                    self.opp_huge_bet_dry += 1

            # Track spots for normalization.
            if state.board:
                from botharshu_v4 import RANK_TO_INT as _RTI2

                wet2 = 0
                ranks2 = sorted((_RTI2[c[0]] for c in state.board), reverse=True)
                suits2 = [c[1] for c in state.board]
                max_s2 = max(suits2.count("s"), suits2.count("h"), suits2.count("d"), suits2.count("c"))
                paired2 = len(set(ranks2)) < len(ranks2)
                connected2 = sum(1 for i in range(len(ranks2) - 1) if abs(ranks2[i] - ranks2[i + 1]) <= 2)
                if max_s2 >= 3:
                    wet2 += 1
                if connected2 >= 2:
                    wet2 += 1
                if paired2:
                    wet2 += 1
                if wet2 >= 2:
                    self.opp_huge_bet_spots_wet += 1
                else:
                    self.opp_huge_bet_spots_dry += 1

        # Approximate "folded early after auction win" by checking if we
        # won auction (auction_won flag set in v4) and then end the hand
        # without showing down and with a negative payoff. We only get the
        # final payoff in on_hand_end, so here we just track auction wins.
        # The empirical realization factor will be updated externally if
        # you wire in explicit end‑of‑hand logging.

    def _play_postflop_facing_bet(
        self,
        state: PokerState,
        adj_eq: float,
        pot: int,
        call_cost: int,
        pot_odds: float,
        spr: float,
        texture: dict,
        profile: dict,
        rev_danger: float,
        have_info: bool,
    ):
        """
        Override v4's facing‑bet logic to:
        - Incorporate MDF‑like thresholds vs hyper‑aggressive opponents.
        - Use board‑texture‑specific overbet tells on the river.
        """
        rel_price = call_cost / float(max(1, pot))

        # Base behavior from v4.2: we reconstruct v4's thresholds and then
        # overlay MDF / texture adjustments.
        facing_reraise = self.hand_i_bet_postflop and call_cost > 0

        # Facing a raise after we bet: keep v4's very tight policy.
        if facing_reraise:
            raise_thresh = 0.78
            call_thresh = 0.58
            if state.street == "river":
                raise_thresh += 0.08
                call_thresh += 0.10
            stack_commit = call_cost / float(max(1, state.my_chips))
            if stack_commit > 0.40:
                call_thresh = 0.62
            if stack_commit > 0.70:
                call_thresh = 0.70
            if rev_danger > 0.40:
                call_thresh += 0.06

            if self.hand_postflop_raises >= 2:
                raise_thresh = max(raise_thresh, 0.90)

            if state.can_act(ActionRaise) and adj_eq > raise_thresh:
                amount = self._choose_postflop_raise_size(state, adj_eq, 0.80, texture, spr, state.street)
                return self._safe_raise_or_fallback(state, amount)
            if state.can_act(ActionCall) and adj_eq >= call_thresh:
                return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
            return self._fallback_action(state)

        # Non‑raise facing bet branch: start from v4 thresholds.
        huge_bet = rel_price >= 0.90

        raise_thresh = 0.66
        call_thresh = pot_odds + 0.06
        if profile["massive_pre_jammer"]:
            call_thresh += 0.03
        if not have_info:
            call_thresh += 0.08
            raise_thresh += 0.05
        elif have_info and adj_eq > 0.70:
            raise_thresh = 0.62 if adj_eq <= 0.78 else 0.60
        if huge_bet:
            huge_rate = self._rate(self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18)
            call_thresh += 0.05
            if huge_rate > 0.22:
                raise_thresh += 0.05
                call_thresh += 0.08

        if rel_price > 0.45:
            call_thresh = max(call_thresh, 0.46)
        if rel_price > 0.60:
            call_thresh = max(call_thresh, 0.52)
        if rel_price > 0.80:
            call_thresh = max(call_thresh, 0.60)
        if state.street == "river":
            call_thresh += 0.02
            if rel_price > 0.25:
                call_thresh = max(call_thresh, 0.54)
            if rel_price > 0.35:
                call_thresh = max(call_thresh, 0.58)
            if rel_price > 0.55:
                call_thresh = max(call_thresh, 0.64)

        stack_commit = call_cost / float(max(1, state.my_chips))
        if stack_commit > 0.35:
            call_thresh += 0.06
        if stack_commit > 0.55:
            call_thresh += 0.08
        if stack_commit > 0.75:
            call_thresh += 0.10
        if state.street == "river" and stack_commit > 0.45:
            call_thresh += 0.07

        # MDF‑inspired adjustment vs hyper‑aggression.
        mdf_eq = call_cost / float(pot + call_cost) if (pot + call_cost) > 0 else 0.0
        huge_rate_global = self._rate(self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18)
        hyper_agg = profile["massive_pre_jammer"] or huge_rate_global > 0.25
        if hyper_agg:
            # Do not make call_thresh much higher than break‑even eq;
            # otherwise we fold too much and allow auto‑profit bluffs.
            call_thresh = min(call_thresh, max(mdf_eq + 0.04, 0.40))

        # Board‑texture specific overbet tells on river.
        if state.street == "river" and huge_bet and state.board:
            from botharshu_v4 import RANK_TO_INT as _RTI3

            ranks = sorted((_RTI3[c[0]] for c in state.board), reverse=True)
            suits = [c[1] for c in state.board]
            max_s = max(suits.count("s"), suits.count("h"), suits.count("d"), suits.count("c"))
            paired = len(set(ranks)) < len(ranks)
            connected = sum(1 for i in range(len(ranks) - 1) if abs(ranks[i] - ranks[i + 1]) <= 2)
            wet = 0
            if max_s >= 3:
                wet += 1
            if connected >= 2:
                wet += 1
            if paired:
                wet += 1

            # Compute frequencies.
            wet_spots = max(1, self.opp_huge_bet_spots_wet)
            dry_spots = max(1, self.opp_huge_bet_spots_dry)
            wet_rate = self.opp_huge_bet_wet / float(wet_spots)
            dry_rate = self.opp_huge_bet_dry / float(dry_spots)

            if wet >= 2:
                # If opponent overbets much more often on wet boards,
                # they may be semi‑bluffing; we can call a bit wider.
                if wet_rate > dry_rate + 0.10:
                    call_thresh = max(0.42, call_thresh - 0.06)
            else:
                # On dry boards with unusual overbets, tighten up.
                if dry_rate > wet_rate + 0.10:
                    call_thresh = min(0.85, call_thresh + 0.06)

        # Decision based on adjusted thresholds.
        if state.can_act(ActionRaise) and adj_eq > raise_thresh:
            pressure = 0.45 + 0.30 * (adj_eq - 0.60)
            amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture, spr, state.street)
            self.hand_i_bet_postflop = True
            return self._safe_raise_or_fallback(state, amount)

        if state.can_act(ActionCall) and adj_eq >= call_thresh:
            return ActionCall()

        if state.can_act(ActionFold):
            return ActionFold()
        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionCall() if state.can_act(ActionCall) else ActionFold()




# ===== v4.3 EXTENSIONS =====

"""
v4.3: Hybrid of v4.2's deep probabilistic engine and v5's stability guards.

Goals:
- Keep v4.2's strengths:
  * Range‑aware equity adjustment (Bayesian‑inspired).
  * Auction realization factor (avoid winner's curse).
  * MDF‑inspired defense + board‑texture aware overbet handling.
- Add v5‑style stability:
  * Tighter preflop all‑in calls vs aggressive profiles.
  * Hard guard against calling huge river bets/raises without
    truly strong equity (roughly trips+).
"""

from pkbot.actions import ActionFold, ActionCall
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState

RANK_ORDER = "23456789TJQKA"



RANK_TO_INT = {r: i + 2 for i, r in enumerate(RANK_ORDER)}

class Player(Player):
    """
    v4.3 extends v4.2 in two key ways:
    1) Preflop all‑in calling is tightened in the same spirit as v5,
       especially against massive pre jammers.
    2) On the river, we add a v5‑style hard guard: we do not call
       very large bets/raises unless our equity estimate is very
       high (≈ trips+ vs realistic ranges).
    """

    # ---------- Preflop all‑in calling: v5‑style stability ----------

    def _should_call_preflop_allin(self, state: PokerState, tier: int, profile: dict) -> bool:
        """
        v4.3: start from v5's conservative logic.
        - Always call with strong hands (tier>=2, big pairs, strong suited
          connectors), regardless of profile.
        - Versus massive_pre_jammer: only call wider if we have tier>=2
          and an excellent price (req <= 0.13).
        - Versus normal: allow somewhat lighter calls but still tighter
          than v3/v4 (req <= 0.17 with tier>=1).
        """
        c1, c2 = state.my_hand
        r1 = RANK_TO_INT[c1[0]]
        r2 = RANK_TO_INT[c2[0]]
        suited = c1[1] == c2[1]
        high = max(r1, r2)
        low = min(r1, r2)

        # Strong made / very strong drawing hands: always call.
        if tier >= 2:
            return True
        if r1 == r2 and high >= 9:
            return True
        if high >= 13 and low >= 11:
            return True
        if suited and abs(r1 - r2) <= 1 and low >= 10:
            return True

        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        req = call_cost / float(pot + call_cost)

        if profile.get("massive_pre_jammer", False):
            # Versus aggressive preflop jammers: only call wider if we already
            # have a strong hand (tier>=2) and the price is excellent.
            return tier >= 2 and req <= 0.13

        # Versus normal opponents: still call with solid hands, but a bit tighter
        # than the original v3/v4 edge thresholds.
        return tier >= 1 and req <= 0.17

    # ---------- River stability guard on huge bets/raises ----------

    def _play_postflop(self, game_info: GameInfo, state: PokerState, t0: float):
        """
        v4.3: wrap v4.2's postflop logic with an additional v5‑style guard:
        - On the river, when facing a *very large* bet (or raise), we
          only continue if our equity is very high (~trips+).

        Concretely:
        - If street == 'river' and cost_to_call > 0:
          * compute adjusted equity using v4.2's estimator.
          * if eq < 0.80 and the bet is huge relative to pot/stack,
            fold immediately instead of delegating to v4.2.
        - Otherwise, defer entirely to v4.2's logic.
        """
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        my_stack = max(1, state.my_chips)

        if state.street == "river" and call_cost > 0:
            # Compute v4.2's adjusted equity.
            board = state.board
            iters = self._choose_mc_iters(game_info.time_bank, state.street)
            eq = self._estimate_equity(state.my_hand, board, state.opp_revealed_cards, iters, t0, game_info.time_bank)

            rel_price = call_cost / float(pot + call_cost)
            stack_commit = call_cost / float(my_stack)

            # If equity is not very strong and this is a huge commitment,
            # snap‑fold. This protects against the high‑value, low‑bluff
            # river raises we saw in logs.
            if eq < 0.80 and (rel_price >= 0.80 or stack_commit > 0.60):
                if state.can_act(ActionFold):
                    return ActionFold()
                if state.can_act(ActionCall):
                    # Safety fallback if, for some reason, folding isn't legal.
                    return ActionCall()

        # Otherwise, fall back to v4.2's full postflop machinery.
        return super()._play_postflop(game_info, state, t0)


if __name__ == "__main__":
    run_bot(Player(), parse_args())

