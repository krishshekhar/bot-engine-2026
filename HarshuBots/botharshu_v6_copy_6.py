"""
v6: v5 + opponent-type adaptation.
- Detects weak opponents (fold to our aggression, rarely raise our bets):
  we bet more with marginal/weak hands when checked to and raise small stabs more.
- Versus strong opponents: keep original tight strategy (fold when we need to pay
  with trash/low equity).
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

        # Opponent model
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

        # Calling station detection
        self.our_postflop_bet_spots = 0
        self.opp_call_our_bet_hits = 0
        self.opp_raise_our_bet_hits = 0

        # v6: track folds to our postflop bets (hand ends before we see call/raise)
        self.our_postflop_bet_count = 0
        self.opp_fold_to_our_postflop_bet_hits = 0
        # v6: recent outcomes when we bet postflop (1=they folded, 0=they called/raised or we lost)
        self._recent_postflop_bet_outcomes: deque = deque(maxlen=14)

        # Preflop raise response: opp fold vs call/raise to our preflop raise (cumulative, NOT per hand)
        self.opp_pf_raise_resp_fold = 0
        self.opp_pf_raise_resp_call = 0
        self._last_preflop_call_cost = 0
        self._hand_pf_resp_counted = False  # per-hand: count at most one response

        # Auction model
        self.opp_bid_exact_samples = deque(maxlen=260)
        self.opp_bid_lower_bounds = deque(maxlen=260)
        self.auction_seen = 0
        self.auction_lost = 0
        self.auction_loss_streak = 0
        self.auction_overbid_pressure_hits = 0

        # Per-hand trackers
        self.hand_i_raised = False
        self.hand_i_bet_postflop = False
        self.hand_my_bid = None
        self.hand_auction_chips = None
        self.hand_auction_processed = False

        # v5: did we pay and win the auction (exclusive info)?
        self.auction_won = False
        # v6fix: did opp pay and win the auction (they see our card)?
        self.opp_won_auction = False

        # Equity cache
        self.equity_cache = {}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.round_num = game_info.round_num
        self.rng.seed((self.round_num * 10007) + (17 if current_state.is_bb else 11))

        self.hand_i_raised = False
        self.hand_i_bet_postflop = False
        self.hand_my_bid = None
        self.hand_auction_chips = None
        self.hand_auction_processed = False
        self.auction_won = False
        self.opp_won_auction = False  # True when OPP paid for auction and sees our card
        self._last_preflop_call_cost = 0
        self._hand_pf_resp_counted = False

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        if self.hand_i_raised:
            self.our_raise_opps += 1
            if len(current_state.opp_revealed_cards) < 2 and current_state.payoff > 0:
                self.opp_fold_to_raise_hits += 1
            # Opp folded to our preflop raise (hand ended preflop, we won without showdown)
            if not self.hand_i_bet_postflop and current_state.payoff > 0 and len(current_state.opp_revealed_cards) < 2:
                self.opp_pf_raise_resp_fold += 1
                self._hand_pf_resp_counted = True
        # v6: when we bet postflop and they folded (we won, no showdown), count it
        if self.hand_i_bet_postflop:
            self.our_postflop_bet_count += 1
            folded = current_state.payoff > 0 and len(current_state.opp_revealed_cards) < 2
            if folded:
                self.opp_fold_to_our_postflop_bet_hits += 1
            self._recent_postflop_bet_outcomes.append(1 if folded else 0)

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

    # ---------- Profile detection ----------

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
        # v6: fold to our postflop bet = they fold trash when we bet (we never see call/raise)
        fold_to_postflop_bet_rate = self._rate(
            self.opp_fold_to_our_postflop_bet_hits, self.our_postflop_bet_count, 0.0
        )
        # v6: recent window — switch back to strong quickly when they stop folding
        recent = self._recent_postflop_bet_outcomes
        recent_fold_rate = (
            sum(recent) / len(recent) if len(recent) >= 5 else None
        )
        # Strong signal they're not weak: recent outcomes show they call/raise (lots of 0s)
        force_strong = recent_fold_rate is not None and recent_fold_rate < 0.22

        # Weak opponent = gives in to our aggression; require recent doesn't say strong
        weak_folding_opponent = (
            not is_calling_station
            and not force_strong
            and (
                # Classic signal: fold to our raises, don't raise our bets much
                (
                    self.our_raise_opps >= 3
                    and self.our_postflop_bet_count >= 3
                    and fold_rate > 0.38
                    and raise_rate < 0.25
                )
                # Key for trash-folders: they often fold to our postflop bets
                or (
                    self.our_postflop_bet_count >= 3
                    and fold_to_postflop_bet_rate > 0.32
                )
            )
            # Stay weak only if recent window still shows folds (or not enough data yet)
            and (recent_fold_rate is None or recent_fold_rate >= 0.22)
        )

        total_pf_resp = self.opp_pf_raise_resp_fold + self.opp_pf_raise_resp_call
        tight_3bet_caller = (
            total_pf_resp >= 8
            and self.opp_pf_raise_resp_call / max(1, total_pf_resp) < 0.20
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
            "weak_folding_opponent": weak_folding_opponent,
            "opp_call_rate": call_rate,
            "opp_raise_rate": raise_rate,
            "auction_loss_rate": (self.auction_lost / float(max(1, self.auction_seen))),
            "auction_loss_streak": self.auction_loss_streak,
            "auction_pressure_rate": self._rate(self.auction_overbid_pressure_hits, self.auction_seen, 0.0),
            "tight_3bet_caller": tight_3bet_caller,
        }

    # ---------- Revealed card analysis ----------

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

    def _play_preflop(self, game_info: GameInfo, state: PokerState):
        """
        v5: same structure as v3, but we *reduce* raise frequency when
        we are the small blind (state.is_bb == False), so we raise a
        bit less often out of position preflop.
        """
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

        # Trash tier (tier 0): fold when facing a cost, otherwise check; never pay for trash.
        if tier == 0:
            if call_cost > 0:
                if state.can_act(ActionFold):
                    return ActionFold()
                if can_check:
                    return ActionCheck()
                return ActionCall()
            if can_check:
                return ActionCheck()
            return self._fallback_action(state)

        # Safety valve: even tier 1 hands should not call large 3-bets/4-bets.
        # When pot_odds are high (opp is raising big), tier 1 hands are dominated.
        if call_cost > 0 and pot_odds > 0.35 and tier <= 1:
            if state.can_act(ActionFold):
                return ActionFold()

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

        # v3: if not BB, bump raises; v5: if not BB, *reduce* raises a bit.
        if not state.is_bb:
            raise_freq -= 0.06

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
            self._last_preflop_call_cost = call_cost
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

    # ---------- Preflop all-in calling (tighter vs aggressive) ----------

    def _should_call_preflop_allin(self, state: PokerState, tier: int, profile: dict) -> bool:
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
            return tier >= 2 and req <= 0.13

        return tier >= 1 and req <= 0.17

    # ---------- Postflop: facing bets (auction-aware, river all-ins) ----------

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
        rel_price = call_cost / float(max(1, pot))
        facing_reraise = self.hand_i_bet_postflop and call_cost > 0

        # FIX B: When OPP won the auction they know one of our hole cards.
        # Their bet is from a position of information advantage - they're betting
        # KNOWING what we have. Discount our equity significantly in this case.
        # The more they've bet across streets, the more credible their confidence.
        if self.opp_won_auction:
            # Opp revealed OUR card. Apply penalty based on bet sizing:
            # large bets = more confident they're ahead = more penalty.
            info_penalty = 0.07
            if rel_price > 0.40:
                info_penalty = 0.10
            if rel_price > 0.70:
                info_penalty = 0.14
            adj_eq = max(0.0, adj_eq - info_penalty)

        # Never pay for trash (pure air). Versus strong opponents use strict 0.16;
        # versus weak opponents who fold to our aggression, relax slightly (0.12) so we
        # don't over-fold to small bets when they might be bluffing.
        air_thresh = 0.12 if profile.get("weak_folding_opponent") else 0.16
        if adj_eq < air_thresh:
            if state.can_act(ActionFold):
                return ActionFold()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return self._fallback_action(state)

        # High-pressure folds after winning auction with bad equity.
        stack_commit = call_cost / float(max(1, state.my_chips))
        if (
            self.auction_won
            and have_info
            and adj_eq < 0.55
            and (rel_price >= 0.60 or stack_commit >= 0.50)
        ):
            if state.can_act(ActionFold):
                return ActionFold()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return self._fallback_action(state)

        if facing_reraise:
            # We bet and villain raised. In this league, these large river
            # raises are almost always strong (top pair+). If the raise is
            # very big and our equity isn't excellent, just fold.
            if (
                state.street == "river"
                and (rel_price >= 0.80 or stack_commit > 0.60)
                and adj_eq < 0.80
                and state.can_act(ActionFold)
            ):
                return ActionFold()

            raise_thresh = 0.85
            call_thresh = 0.54

            if self.auction_won and rel_price > 0.75:
                if state.street == "flop":
                    call_thresh = max(call_thresh, 0.66)
                elif state.street == "turn":
                    call_thresh = max(call_thresh, 0.70)
                raise_thresh = max(raise_thresh, 0.88)
            if state.street == "river":
                raise_thresh += 0.08
                call_thresh += 0.10
            if stack_commit > 0.40:
                call_thresh = 0.60
            if stack_commit > 0.70:
                call_thresh = 0.66
            if rev_danger > 0.40:
                call_thresh += 0.06
            # Don't stack off with one pair when they've shown strength (reraise)
            if stack_commit > 0.35 and adj_eq < 0.66:
                call_thresh = 0.99
            if stack_commit > 0.50 and adj_eq < 0.72:
                raise_thresh = 0.99
            # When opp won the auction (no info for us), their reraise is even more
            # credible because they know one of our cards. Be extra conservative.
            if not have_info:
                raise_thresh += 0.08

            if state.can_act(ActionRaise) and adj_eq > raise_thresh:
                amount = self._choose_postflop_raise_size(
                    state, adj_eq, 0.80, texture, spr, state.street
                )
                return self._safe_raise_or_fallback(state, amount)
            if state.can_act(ActionCall) and adj_eq >= call_thresh:
                return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
            return self._fallback_action(state)

        small_stab = rel_price <= 0.16
        huge_bet = rel_price >= 0.90

        # Versus weak opponents: raise small stabs more often with marginal hands.
        small_stab_eq_min = 0.38 if profile.get("weak_folding_opponent") else 0.45
        small_stab_punish_freq = 0.68 if profile.get("weak_folding_opponent") else 0.55
        if small_stab and state.can_act(ActionRaise) and adj_eq > small_stab_eq_min:
            punish_gate = profile["fold_rate"] > 0.40 or self._rate(
                self.opp_small_stab_hits, self.opp_post_bet_spots, 0.22
            ) > 0.35
            if punish_gate and self.rng.random() < small_stab_punish_freq:
                pressure = self._clip(0.40 + 0.20 * (adj_eq - 0.45), 0.30, 0.70)
                amount = self._choose_postflop_raise_size(
                    state, adj_eq, pressure, texture, spr, state.street
                )
                self.hand_i_bet_postflop = True
                return self._safe_raise_or_fallback(state, amount)

        raise_thresh = 0.66
        call_thresh = pot_odds + 0.06
        if profile["massive_pre_jammer"]:
            call_thresh += 0.03
        if not have_info:
            call_thresh += 0.08
            raise_thresh += 0.05
            # On paired/connected boards without any revealed card info, we can't
            # distinguish if opp hit trips/straight. Raise threshold goes up.
            if texture["wetness"] >= 2:
                raise_thresh += 0.07
        # FIX A: When we have info (we revealed opp's card), do NOT lower raise_thresh.
        # Knowing opp's one card still leaves their other card unknown; that hidden
        # card can make trips/flush/straight. Keep threshold high to prevent
        # all-in raises with one pair based on "safe" revealed low card.
        # We only get a small bonus for very high equity (>0.82) to be slightly
        # more aggressive when we're genuinely strong.
        elif have_info and adj_eq > 0.82:
            raise_thresh = 0.76
        if huge_bet:
            huge_rate = self._rate(
                self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18
            )
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

        if stack_commit > 0.35:
            call_thresh += 0.06
        if stack_commit > 0.55:
            call_thresh += 0.08
        if stack_commit > 0.75:
            call_thresh += 0.10
        if state.street == "river" and stack_commit > 0.45:
            call_thresh += 0.07

        # Extra river all-in guards:
        if state.street == "river":
            # 1) League is very value-heavy on big river bets; only continue
            #    if our equity is very strong (~trips+ equivalent).
            if (rel_price >= 0.80 or stack_commit > 0.60) and adj_eq < 0.80:
                if state.can_act(ActionFold):
                    return ActionFold()

            # 2) Don't call large river overbets with marginal one pair / second pair
            if rel_price >= 0.50 and adj_eq < 0.55 and state.can_act(ActionFold):
                return ActionFold()

            # 3) Versus aggressive profiles, be even more conservative.
            huge_rate = self._rate(
                self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18
            )
            aggressive = profile["massive_pre_jammer"] or huge_rate > 0.22
            if aggressive and stack_commit > 0.50 and adj_eq < 0.72:
                if state.can_act(ActionFold):
                    return ActionFold()

        if state.can_act(ActionRaise) and adj_eq > raise_thresh:
            pressure = 0.45 + 0.30 * (adj_eq - 0.60)
            amount = self._choose_postflop_raise_size(
                state, adj_eq, pressure, texture, spr, state.street
            )
            self.hand_i_bet_postflop = True
            return self._safe_raise_or_fallback(state, amount)

        if state.can_act(ActionCall) and adj_eq >= call_thresh:
            return ActionCall()
        if (
            state.can_act(ActionCall)
            and adj_eq >= 0.38
            and pot_odds < 0.28
            and (call_thresh - adj_eq) < 0.08
        ):
            return ActionCall()

        if state.can_act(ActionFold):
            return ActionFold()
        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionCall() if state.can_act(ActionCall) else ActionFold()

    def _play_postflop_checked_to(self, state, adj_eq, pot, spr, texture, profile, rev_danger):
        if not state.can_act(ActionRaise):
            return ActionCheck() if state.can_act(ActionCheck) else self._fallback_action(state)

        is_calling_station = profile["calling_station"]
        fold_rate = profile["fold_rate"]
        weak_opp = profile.get("weak_folding_opponent", False)
        # Versus weak opponents who fold to our aggression: bet more with marginal hands.
        value_thresh = 0.50 if weak_opp else (0.54 if state.street == "flop" else 0.56)

        if is_calling_station:
            if adj_eq > value_thresh + 0.04:
                pressure = self._clip(0.40 + 0.30 * (adj_eq - 0.55), 0.30, 0.75)
                amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture, spr, state.street)
                self.hand_i_bet_postflop = True
                return self._safe_raise_or_fallback(state, amount)
            return ActionCheck() if state.can_act(ActionCheck) else self._fallback_action(state)

        value_bet = adj_eq > value_thresh
        # Versus weak opponents: more frequent bluffs and pressure with weak hands.
        bluff_fold_min = 0.40 if weak_opp else 0.56
        bluff_freq = 0.22 if weak_opp else 0.12
        pressure_fold_min = 0.38 if weak_opp else 0.48
        pressure_freq = 0.24 if weak_opp else 0.14
        semi_fold_min = 0.44 if weak_opp else 0.52
        semi_freq = 0.18 if weak_opp else 0.10

        bluff_ok = (
            adj_eq < 0.42
            and texture["wetness"] == 0
            and fold_rate > bluff_fold_min
            and rev_danger < 0.25
            and self.rng.random() < bluff_freq
        )
        pressure_ok = (
            fold_rate > pressure_fold_min
            and rev_danger < 0.30
            and adj_eq > (0.36 if weak_opp else 0.40)
            and self.rng.random() < pressure_freq
        )
        semi_bluff_ok = (
            state.street == "turn"
            and (0.34 if weak_opp else 0.38) < adj_eq < 0.50
            and texture["wetness"] >= 1
            and fold_rate > semi_fold_min
            and rev_danger < 0.30
            and self.rng.random() < semi_freq
        )

        if value_bet or bluff_ok or pressure_ok or semi_bluff_ok:
            if bluff_ok:
                pressure = 0.30
            elif pressure_ok and not value_bet:
                pressure = 0.35
            elif semi_bluff_ok:
                pressure = 0.32
            else:
                pressure = self._clip(0.40 + 0.30 * max(0.0, adj_eq - 0.52), 0.30, 0.75)
                if state.street == "river" and adj_eq > 0.65:
                    pressure = self._clip(pressure + 0.05, 0.35, 0.80)
            amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture, spr, state.street)
            self.hand_i_bet_postflop = True
            return self._safe_raise_or_fallback(state, amount)

        if state.can_act(ActionCheck):
            return ActionCheck()
        return self._fallback_action(state)

    # ---------- Auction ----------

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

        self.hand_my_bid = bid
        return bid

    # ---------- Opponent model updates ----------

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

            # UDF1 raised preflop and opp responded on same street (call/raise only; fold counted in on_hand_end)
            if state.street == 'pre-flop' and self.hand_i_raised and not self._hand_pf_resp_counted:
                if state.cost_to_call > 0 or state.opp_wager > self._last_preflop_call_cost:
                    self.opp_pf_raise_resp_call += 1
                    self._hand_pf_resp_counted = True

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

            self.auction_won = False

            if d_my > 0 and d_opp == 0:
                self.opp_bid_exact_samples.append(d_my)
                self.auction_seen += 1
                self.auction_loss_streak = 0
                self.auction_won = True
            elif d_opp > 0 and d_my == 0:
                self.opp_bid_lower_bounds.append(max(1, self.hand_my_bid + 1))
                self.auction_seen += 1
                self.auction_lost += 1
                self.auction_loss_streak += 1
                self.opp_won_auction = True  # Opp paid for and sees our card
                if d_opp >= max(120, int(0.18 * max(1, state.pot))):
                    self.auction_overbid_pressure_hits += 1
            elif d_my > 0 and d_opp > 0:
                self.opp_bid_exact_samples.append(d_opp)
                self.auction_seen += 1
                self.auction_loss_streak = 0

            self.hand_auction_processed = True

    # ---------- Action safety and fallback ----------

    def _safe_raise_or_fallback(self, state: PokerState, amount: int):
        if state.can_act(ActionRaise):
            min_r, max_r = state.raise_bounds
            amt = int(self._clip(amount, min_r, max_r))
            if min_r <= amt <= max_r:
                self.hand_i_raised = True
                return ActionRaise(amt)
        return self._fallback_action(state)

    def _safe_bid(self, state: PokerState, amount: int):
        if state.can_act(ActionBid):
            # Auction bids must respect stack size. When we're all-in (my_chips == 0),
            # the only legal bid is 0; attempting 1 causes an illegal ActionBid.
            max_bid = max(0, int(state.my_chips))
            amt = int(self._clip(int(amount), 0, max_bid))
            if amt == 0 and max_bid > 0:
                # Prefer making a live (non-zero) bid when we can afford it.
                amt = 1
            return ActionBid(amt)
        return self._fallback_action(state)

    def _fallback_action(self, state: PokerState):
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionBid):
            # Safe auction fallback: 0 is always legal (and required when all-in).
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

    def _choose_preflop_raise_size(self, state: PokerState, tier: int, profile: dict) -> int:
        min_r, max_r = state.raise_bounds
        span = max(0, max_r - min_r)
        fold_rate = profile["fold_rate"]
        pot = max(1, state.pot)
        call_cost = max(0, state.cost_to_call)

        # When opp only calls raises with premiums (AA/KK/QQ), size down to 2.5x
        # to preserve steal equity while capping damage when they do call.
        if profile.get("tight_3bet_caller") and call_cost > 0:
            target = int(call_cost * 2.5)
            return int(self._clip(target, min_r, max_r))

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
        # Premiums: don't open-shove when pot is just blinds — use standard raise to get action
        BIG_BLIND = 20
        if tier >= 3 and pot <= 60:
            standard_open = 4 * BIG_BLIND
            target = min(target, max(min_r, standard_open))
        return int(self._clip(target, min_r, max_r))

    def _choose_postflop_raise_size(self, state: PokerState, eq: float, pressure: float, texture: dict, spr: float, street: str) -> int:
        min_r, max_r = state.raise_bounds
        pot = max(1, state.pot)

        base_multiplier = self._clip(pressure, 0.25, 0.85)
        target_bet = int(pot * base_multiplier)

        if eq > 0.75:
            target_bet = int(pot * self._clip(pressure + 0.12, 0.40, 0.95))
        if spr < 1.0 and eq > 0.60:
            target_bet = max(target_bet, int(0.80 * state.my_chips))
        if street == "river" and eq > 0.65:
            target_bet = max(target_bet, int(pot * 0.72))

        wet_penalty = -int(0.08 * pot) if texture["wetness"] >= 2 and eq < 0.58 else 0
        target_bet += wet_penalty

        target_bet += self.rng.randint(-int(0.05 * pot) - 1, int(0.05 * pot) + 1)
        return int(self._clip(target_bet, min_r, max_r))

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

    def _estimate_equity_vs_revealed(
        self,
        my_hand: list[str],
        board: list[str],
        revealed: str,
        iters: int,
        t0: float,
        time_bank: float,
    ) -> float:
        return self._estimate_equity(my_hand, board, [revealed], iters, t0, time_bank)

    def _estimate_equity_vs_revealed_mix(
        self,
        my_hand: list[str],
        board: list[str],
        iters: int,
        t0: float,
        time_bank: float,
    ) -> float:
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

    # ---------- Utility helpers ----------

    def _preflop_hand_key(self, c1: str, c2: str) -> str:
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        if r1 == r2:
            return r1 + r2
        hi, lo = (r1, r2) if RANK_TO_INT[r1] >= RANK_TO_INT[r2] else (r2, r1)
        suited = "s" if s1 == s2 else "o"
        return hi + lo + suited

    def _preflop_tier(self, hand_key: str) -> int:
        if hand_key in PREMIUM:
            return 3
        if hand_key in STRONG:
            return 2
        if hand_key in PLAYABLE:
            return 1
        return 0

    def _should_rare_preflop_fold(self, state: PokerState, tier: int, profile: dict, pot_odds: float) -> bool:
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

    def _board_texture(self, board: list[str]) -> dict:
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

    def _predict_opp_bid(self, opp_stack: int, pot: int) -> int:
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

    def _opp_fold_to_raise_rate(self) -> float:
        return self._rate(self.opp_fold_to_raise_hits, self.our_raise_opps, 0.35)

    def _choose_mc_iters(self, time_bank: float, street: str) -> int:
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

    def _per_decision_time_budget(self, time_bank: float) -> float:
        if time_bank < 1.5:
            return 0.012
        if time_bank < 4.0:
            return 0.020
        if time_bank < 8.0:
            return 0.033
        return 0.050

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

