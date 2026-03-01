'''
Competition poker bot for Sneak Peek Hold'em.
'''
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import random
import time
from collections import deque
import eval7


RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i + 2 for i, r in enumerate(RANK_ORDER)}

# Preflop table (compact top-range style buckets).
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
    '''
    Competition-ready pokerbot.
    '''

    def __init__(self) -> None:
        self.rng = random.Random()
        self.round_num = 0

        # Opponent model counters
        self.opp_vpip_opps = 0
        self.opp_vpip_hits = 0
        self.opp_pfr_opps = 0
        self.opp_pfr_hits = 0
        self.opp_aggr_opps = 0
        self.opp_aggr_hits = 0
        self.opp_fold_to_raise_opps = 0
        self.opp_fold_to_raise_hits = 0
        self.opp_checkraise_opps = 0
        self.opp_checkraise_hits = 0
        self.opp_bluff_showdowns = 0
        self.opp_bluff_hits = 0
        self.opp_small_stab_opps = 0
        self.opp_small_stab_hits = 0
        self.opp_huge_bet_opps = 0
        self.opp_huge_bet_hits = 0
        self.opp_massive_pre_raises_seen = 0
        self.opp_pre_raise_spots = 0
        self.opp_small_pre_raise_spots = 0
        self.opp_small_pre_raise_hits = 0

        # Auction model
        self.opp_bid_exact_samples = deque(maxlen=200)
        self.opp_bid_lower_bounds = deque(maxlen=200)
        self.my_bid_history = deque(maxlen=200)
        self.auction_total = 0
        self.auction_ties = 0
        self.auction_won = 0

        # Per hand tracking
        self.hand_last_street = None
        self.hand_opp_raised_post = False
        self.hand_opp_checked_street = {}
        self.hand_opp_raised_after_check = False
        self.hand_i_raised = False
        self.hand_raise_street = None
        self.hand_my_bid = None
        self.hand_auction_chips = None
        self.hand_auction_processed = False
        self.hand_seen_call_to_raise = False

        # Caches
        self.equity_cache = {}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.round_num = game_info.round_num
        self.rng.seed((self.round_num * 10007) + (17 if current_state.is_bb else 11))

        self.hand_last_street = current_state.street
        self.hand_opp_raised_post = False
        self.hand_opp_checked_street = {}
        self.hand_opp_raised_after_check = False
        self.hand_i_raised = False
        self.hand_raise_street = None
        self.hand_my_bid = None
        self.hand_auction_chips = None
        self.hand_auction_processed = False
        self.hand_seen_call_to_raise = False

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        # Opponent fold-to-raise proxy: if we raised and hand ended without showdown.
        if self.hand_i_raised:
            self.opp_fold_to_raise_opps += 1
            if len(current_state.opp_revealed_cards) < 2 and current_state.payoff > 0:
                self.opp_fold_to_raise_hits += 1

        # Bluff proxy at showdown: opponent took aggressive line with weak-ish shown hand.
        if len(current_state.opp_revealed_cards) == 2 and self.hand_opp_raised_post:
            self.opp_bluff_showdowns += 1
            if self._shown_hand_is_weak(current_state.board, current_state.opp_revealed_cards):
                self.opp_bluff_hits += 1

    def get_move(
        self,
        game_info: GameInfo,
        current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        t0 = time.perf_counter()
        self._update_live_opponent_model(current_state)

        if current_state.street == "auction":
            bid = self._choose_auction_bid(game_info, current_state, t0)
            return self._safe_bid(current_state, bid)

        if current_state.street == "pre-flop":
            return self._play_preflop(game_info, current_state, t0)

        return self._play_postflop(game_info, current_state, t0)

    # ---------- Strategy Core ----------

    def _play_preflop(self, game_info: GameInfo, state: PokerState, t0: float):
        hand_key = self._preflop_hand_key(state.my_hand[0], state.my_hand[1])
        tier = self._preflop_tier(hand_key)

        bankroll_pressure = self._bankroll_factor(game_info.bankroll, game_info.round_num)
        opp_fold_raise = self._rate(self.opp_fold_to_raise_hits, self.opp_fold_to_raise_opps, 0.35)
        loosen_factor = 0.08 if opp_fold_raise > 0.5 else 0.0
        tighten_factor = 0.08 if bankroll_pressure < -0.4 else 0.0

        can_raise = state.can_act(ActionRaise)
        can_call = state.can_act(ActionCall)
        can_check = state.can_act(ActionCheck)

        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(pot + call_cost) if call_cost > 0 else 0.0
        massive_pre_rate = self._rate(self.opp_massive_pre_raises_seen, self.opp_pre_raise_spots, 0.10)
        small_pre_rate = self._rate(self.opp_small_pre_raise_hits, self.opp_small_pre_raise_spots, 0.25)

        # Base frequencies by tier.
        if tier == 3:  # premium
            raise_freq = 0.94
            call_freq = 0.05
        elif tier == 2:  # strong
            raise_freq = 0.82
            call_freq = 0.16
        elif tier == 1:  # playable
            raise_freq = 0.66
            call_freq = 0.26
        else:  # marginal/trash
            raise_freq = 0.42
            call_freq = 0.16

        # Position and pressure adjustments.
        if not state.is_bb:
            raise_freq += 0.06
        if call_cost > 0.10 * state.my_chips:
            raise_freq -= 0.10
            call_freq -= 0.06
        if call_cost > 0.25 * state.my_chips:
            raise_freq -= 0.15
            call_freq -= 0.12
        if call_cost > 0.42 * state.my_chips:
            raise_freq -= 0.18
            call_freq -= 0.18
        raise_freq += loosen_factor - tighten_factor
        call_freq += (loosen_factor * 0.5) - (tighten_factor * 0.8)

        # Anti-bully adaptation: if opponent frequently uses massive preflop sizing,
        # tighten trash folds but continue with robust tiers to avoid getting run over.
        if massive_pre_rate > 0.18 and call_cost > 0.22 * state.my_chips:
            if tier == 0:
                call_freq *= 0.45
                raise_freq *= 0.60
            elif tier == 1:
                call_freq *= 0.78
                raise_freq *= 0.78
            else:
                call_freq *= 1.14

        # Min-raise-heavy opponents (e.g., frequent 40/70 opens) can trap
        # oversized 3-bets; defend wider in position and keep trash less explosive.
        if small_pre_rate > 0.46 and call_cost <= 90:
            if tier == 0:
                call_freq += 0.14
                raise_freq -= 0.18
            elif tier == 1:
                call_freq += 0.08
                raise_freq -= 0.10
            else:
                raise_freq += 0.06

        # Pot-odds protection for weak hands.
        if tier == 0 and pot_odds > 0.24:
            call_freq *= 0.35
        if tier <= 1 and call_cost > 200:
            call_freq *= 0.7

        raise_freq = self._clip(raise_freq, 0.0, 1.0)
        call_freq = self._clip(call_freq, 0.0, 0.95)

        roll = self.rng.random()
        if can_raise and roll < raise_freq:
            amount = self._choose_preflop_raise_size(state, tier, opp_fold_raise, small_pre_rate)
            return self._safe_raise_or_fallback(state, amount)

        if can_call and roll < raise_freq + call_freq:
            return ActionCall()

        if can_check:
            return ActionCheck()
        if can_call and (tier >= 1 or pot_odds < 0.18):
            return ActionCall()
        return ActionFold()

    def _play_postflop(self, game_info: GameInfo, state: PokerState, t0: float):
        board = state.board
        opp_revealed = state.opp_revealed_cards

        iters = self._choose_mc_iters(game_info.time_bank, state.street)
        eq = self._estimate_equity(state.my_hand, board, opp_revealed, iters, t0, game_info.time_bank)

        # If we won auction and have one card, refine with weighted range estimate.
        if len(opp_revealed) == 1:
            eq_reveal = self._estimate_equity_vs_revealed(state.my_hand, board, opp_revealed[0], max(60, iters // 2), t0, game_info.time_bank)
            eq = 0.62 * eq_reveal + 0.38 * eq

        texture = self._board_texture(board)
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(pot + call_cost) if call_cost > 0 else 0.0
        draw_bonus = 0.04 if texture["wetness"] >= 2 else 0.0
        implied = 0.02 if len(board) < 5 and call_cost <= 0.22 * state.my_chips else 0.0
        adj_eq = self._clip(eq + draw_bonus + implied, 0.0, 1.0)

        opp_aggr = self._rate(self.opp_aggr_hits, self.opp_aggr_opps, 0.45)
        opp_bluff = self._rate(self.opp_bluff_hits, self.opp_bluff_showdowns, 0.18)
        opp_fold_raise = self._rate(self.opp_fold_to_raise_hits, self.opp_fold_to_raise_opps, 0.35)

        # Facing a bet: call/fold/raise decision.
        if call_cost > 0:
            rel_price = call_cost / float(max(1, pot))
            small_stab = rel_price <= 0.18
            huge_bet = rel_price >= 0.90

            # Punish tiny probes from passive opponents with more raises.
            if (
                small_stab
                and state.can_act(ActionRaise)
                and (
                    opp_fold_raise > 0.36
                    or self._rate(self.opp_small_stab_hits, self.opp_small_stab_opps, 0.25) > 0.33
                )
                and self.rng.random() < 0.74
            ):
                punish = self._clip(0.72 + 0.20 * (adj_eq - 0.45), 0.52, 0.95)
                amount = self._choose_postflop_raise_size(state, adj_eq, punish, texture)
                return self._safe_raise_or_fallback(state, amount)

            raise_thresh = 0.63 - (0.05 if opp_bluff < 0.14 else 0.0)
            call_thresh = pot_odds + 0.02 - (0.03 if opp_bluff > 0.24 else 0.0)

            # Tighten versus repeated huge overbets unless we have robust equity.
            if huge_bet:
                huge_rate = self._rate(self.opp_huge_bet_hits, self.opp_huge_bet_opps, 0.18)
                if huge_rate > 0.22:
                    raise_thresh += 0.04
                    call_thresh += 0.06
                if adj_eq < call_thresh and state.can_act(ActionFold):
                    return ActionFold()

            if state.can_act(ActionRaise) and adj_eq > raise_thresh:
                pressure = 0.58 + 0.27 * (adj_eq - 0.63) + (0.05 if opp_aggr > 0.55 else 0.0)
                amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture)
                return self._safe_raise_or_fallback(state, amount)

            if state.can_act(ActionCall) and adj_eq >= call_thresh:
                return ActionCall()

            if state.can_act(ActionFold):
                return ActionFold()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionCall() if state.can_act(ActionCall) else ActionFold()

        # No bet to us: check vs value/bluff betting.
        if state.can_act(ActionRaise):
            value_bet = adj_eq > (0.55 if len(board) == 3 else 0.58)
            bluff_spot = (adj_eq < 0.46 and texture["wetness"] == 0 and opp_fold_raise > 0.48)
            mixed_bluff = bluff_spot and self.rng.random() < (0.18 + 0.20 * (opp_fold_raise - 0.48))
            pressure_spot = opp_fold_raise > 0.40 and self.rng.random() < 0.35

            if value_bet or mixed_bluff or pressure_spot:
                pressure = 0.52 + 0.34 * max(0.0, adj_eq - 0.52)
                if mixed_bluff:
                    pressure = 0.42
                elif pressure_spot and not value_bet:
                    pressure = 0.60
                amount = self._choose_postflop_raise_size(state, adj_eq, pressure, texture)
                return self._safe_raise_or_fallback(state, amount)

        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        return ActionFold()

    # ---------- Auction ----------

    def _choose_auction_bid(self, game_info: GameInfo, state: PokerState, t0: float) -> int:
        self.hand_auction_chips = (state.my_chips, state.opp_chips)

        # EV without extra info and with one-card reveal proxy.
        iters = self._choose_mc_iters(game_info.time_bank, "auction")
        eq_without = self._estimate_equity(state.my_hand, state.board, [], iters, t0, game_info.time_bank)
        eq_with = self._estimate_equity_vs_revealed_mix(state.my_hand, state.board, max(80, iters // 2), t0, game_info.time_bank)
        value_of_info = max(0.0, eq_with - eq_without)

        pot = max(1, state.pot)
        bankroll_factor = self._bankroll_factor(game_info.bankroll, game_info.round_num)
        stack_ratio = state.my_chips / float(max(1, state.my_chips + state.opp_chips))

        # Opponent bid prediction from censored + exact samples.
        pred_opp_bid = self._predict_opp_bid(state.opp_chips, pot)
        pred_norm = pred_opp_bid / float(max(1, state.opp_chips))

        # Risk-adjusted target in second-price auction.
        base = value_of_info * pot
        if eq_without > 0.68:  # very strong made/equity hand: info less valuable.
            base *= 0.65
        elif 0.42 <= eq_without <= 0.58:  # marginal spots benefit more from info.
            base *= 1.24

        if stack_ratio < 0.35:
            base *= 0.72
        if bankroll_factor > 0.45:
            base *= 0.86
        elif bankroll_factor < -0.35:
            base *= 1.08

        aggr = self._rate(self.opp_aggr_hits, self.opp_aggr_opps, 0.45)
        if aggr > 0.56:
            base *= 1.10

        # Opponent with tiny auction bids is best countered by cheap overcalls
        # to win information at minimal second-price cost.
        if self._is_micro_bidder(state.opp_chips, pot, pred_opp_bid):
            if eq_without > 0.70:
                target = 0 if self.rng.random() < 0.55 else min(state.my_chips, 3)
            else:
                tiny_cover = pred_opp_bid + (2 if self.rng.random() < 0.75 else 1)
                if 0.40 <= eq_without <= 0.62:
                    tiny_cover += 1
                target = min(state.my_chips, max(0, tiny_cover))
        # Mid fixed bidders (often around 9-12): pay just above median frequently
        # instead of overbidding by pot-based logic.
        elif self._is_fixed_mid_bidder(state.opp_chips, pot, pred_opp_bid):
            if eq_without > 0.72 and self.rng.random() < 0.40:
                target = 0
            else:
                target = min(state.my_chips, max(0, pred_opp_bid + 1))
        # Opponent with fixed high bids is best countered by underbidding and
        # forcing them to overpay for reveals while we attack postflop.
        elif self._is_overbidding_opponent(state.opp_chips, pot, pred_opp_bid):
            block = min(state.my_chips, max(0, min(pred_opp_bid // 3, int(0.16 * pot))))
            if self.rng.random() < 0.22 and 0.40 <= eq_without <= 0.62 and state.my_chips > 700:
                # Occasional snipe near their modal levels.
                target = pred_opp_bid + int(self.rng.uniform(-12, 18))
            else:
                target = block if self.rng.random() > 0.12 else 0
        else:
            # Blend toward opponent prediction to improve win chance without overpaying.
            target = 0.55 * base + 0.45 * (pred_opp_bid * (0.92 + 0.20 * value_of_info))

        # Nash-like mixed strategy around target to reduce predictability.
        sigma = max(8.0, 0.18 * target + 6.0)
        noisy = target + self.rng.gauss(0.0, sigma)

        # Hard caps for robustness and bankroll protection.
        max_reasonable = min(state.my_chips, int(0.42 * (pot + state.my_chips)))
        min_reasonable = 0
        bid = int(self._clip(noisy, min_reasonable, max_reasonable))

        # Occasional low-end randomization to avoid deterministic tells.
        if self.rng.random() < 0.07:
            bid = int(0.6 * bid)

        self.hand_my_bid = bid
        self.my_bid_history.append(bid)
        return bid

    # ---------- Helpers ----------

    def _update_live_opponent_model(self, state: PokerState) -> None:
        # Track voluntary preflop investment and preflop raise opportunities.
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
                # Massive preflop raise profile (observed in some bully bots).
                if state.cost_to_call >= 850 or state.opp_wager >= 900:
                    self.opp_massive_pre_raises_seen += 1
                self.opp_small_pre_raise_spots += 1
                if state.opp_wager <= 120 or state.cost_to_call <= 80:
                    self.opp_small_pre_raise_hits += 1

        # Track postflop aggression proxy.
        if state.street in ("flop", "turn", "river"):
            self.opp_aggr_opps += 1
            if state.cost_to_call > 0:
                self.opp_aggr_hits += 1
                self.hand_opp_raised_post = True
                rel = state.cost_to_call / float(max(1, state.pot))
                self.opp_small_stab_opps += 1
                if rel <= 0.18:
                    self.opp_small_stab_hits += 1
                self.opp_huge_bet_opps += 1
                if rel >= 0.90:
                    self.opp_huge_bet_hits += 1

            # Check-raise proxy using observed no-cost point followed by facing bet.
            was_checked = self.hand_opp_checked_street.get(state.street, False)
            if state.cost_to_call == 0:
                self.hand_opp_checked_street[state.street] = True
            elif was_checked:
                self.opp_checkraise_opps += 1
                self.opp_checkraise_hits += 1
                self.hand_opp_raised_after_check = True

        # Process auction result once after auction transition.
        if (
            state.street == "flop"
            and not state.can_act(ActionBid)
            and self.hand_my_bid is not None
            and not self.hand_auction_processed
            and self.hand_auction_chips is not None
        ):
            before_my, before_opp = self.hand_auction_chips
            d_my = max(0, before_my - state.my_chips)
            d_opp = max(0, before_opp - state.opp_chips)
            self.auction_total += 1

            # Tie: both paid own bids (equal bids by rules).
            if d_my > 0 and d_opp > 0:
                self.auction_ties += 1
                self.opp_bid_exact_samples.append(d_opp)
            elif d_my > 0 and d_opp == 0:
                # We won and paid opponent bid exactly.
                self.auction_won += 1
                self.opp_bid_exact_samples.append(d_my)
            elif d_opp > 0 and d_my == 0:
                # Opponent won, so opp bid strictly greater than our bid.
                self.opp_bid_lower_bounds.append(self.hand_my_bid + 1)

            self.hand_auction_processed = True

        self.hand_last_street = state.street

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

    def _choose_preflop_raise_size(self, state: PokerState, tier: int, opp_fold_raise: float, small_pre_rate: float) -> int:
        min_r, max_r = state.raise_bounds
        span = max(0, max_r - min_r)
        if tier >= 3:
            frac = 0.92
        elif tier == 2:
            frac = 0.84
        elif tier == 1:
            frac = 0.72
        else:
            frac = 0.60
        if opp_fold_raise > 0.44:
            frac += 0.10
        if small_pre_rate > 0.46 and tier <= 1:
            frac -= 0.14
        frac += self.rng.uniform(-0.05, 0.05)
        target = min_r + int(span * self._clip(frac, 0.05, 0.95))
        return int(self._clip(target, min_r, max_r))

    def _choose_postflop_raise_size(self, state: PokerState, eq: float, pressure: float, texture: dict) -> int:
        min_r, max_r = state.raise_bounds
        span = max(0, max_r - min_r)

        wet_penalty = 0.08 if texture["wetness"] >= 2 and eq < 0.6 else 0.0
        frac = self._clip(pressure + 0.20 * (eq - 0.5) - wet_penalty, 0.16, 0.96)
        frac += self.rng.uniform(-0.07, 0.07)

        target = min_r + int(span * self._clip(frac, 0.05, 0.99))
        return int(self._clip(target, min_r, max_r))

    def _safe_raise_or_fallback(self, state: PokerState, amount: int):
        if state.can_act(ActionRaise):
            min_r, max_r = state.raise_bounds
            amt = int(self._clip(amount, min_r, max_r))
            if min_r <= amt <= max_r:
                self.hand_i_raised = True
                self.hand_raise_street = state.street
                return ActionRaise(amt)
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionCheck):
            return ActionCheck()
        return ActionFold()

    def _safe_bid(self, state: PokerState, amount: int):
        if state.can_act(ActionBid):
            amt = int(self._clip(amount, 0, state.my_chips))
            return ActionBid(amt)
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        return ActionFold()

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

    def _estimate_equity(
        self,
        my_hand: list[str],
        board: list[str],
        opp_revealed: list[str],
        iters: int,
        t0: float,
        time_bank: float,
    ) -> float:
        key = (tuple(sorted(my_hand)), tuple(board), tuple(sorted(opp_revealed)), len(board), iters // 50)
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

        for _ in range(max(20, iters)):
            if time.perf_counter() - t0 > max_runtime:
                break

            if len(revealed_cards) == 2:
                opp_cards = revealed_cards
                draw_count = need_board
            elif len(revealed_cards) == 1:
                draw_count = need_board + 1
                drawn = self.rng.sample(rem, draw_count)
                opp_cards = [revealed_cards[0], drawn[0]]
                board_draw = drawn[1:]
            else:
                draw_count = need_board + 2
                drawn = self.rng.sample(rem, draw_count)
                opp_cards = [drawn[0], drawn[1]]
                board_draw = drawn[2:]

            if len(revealed_cards) == 2:
                board_draw = self.rng.sample(rem, need_board)

            full_board = board_cards + board_draw
            my_score = eval7.evaluate(my_cards + full_board)
            opp_score = eval7.evaluate(opp_cards + full_board)

            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
            n += 1

        eq = wins / n if n > 0 else 0.5
        if len(self.equity_cache) > 6000:
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
        # Value of information proxy: expected equity if we get one random opponent card.
        known = set(my_hand + board)
        candidates = [r + s for r in RANK_ORDER for s in "shdc" if (r + s) not in known]
        if not candidates:
            return 0.5

        samples = min(12, len(candidates))
        chosen = self.rng.sample(candidates, samples)
        total = 0.0
        for c in chosen:
            total += self._estimate_equity_vs_revealed(my_hand, board, c, max(20, iters // samples), t0, time_bank)
        return total / float(samples)

    def _choose_mc_iters(self, time_bank: float, street: str) -> int:
        if time_bank < 3.0:
            base = 70
        elif time_bank < 7.0:
            base = 110
        elif time_bank < 12.0:
            base = 170
        else:
            base = 240

        if street == "river":
            return int(base * 0.70)
        if street == "turn":
            return int(base * 0.85)
        if street == "flop":
            return int(base * 1.10)
        if street == "auction":
            return int(base * 0.95)
        return base

    def _shown_hand_is_weak(self, board: list[str], opp_cards: list[str]) -> bool:
        try:
            cards = [eval7.Card(c) for c in (board + opp_cards)]
            score = eval7.evaluate(cards)
            # Weak proxy threshold (high card / low pair-ish region).
            return score < 3400
        except Exception:
            return False

    def _predict_opp_bid(self, opp_stack: int, pot: int) -> int:
        if self.opp_bid_exact_samples:
            mean_exact = sum(self.opp_bid_exact_samples) / float(len(self.opp_bid_exact_samples))
        else:
            mean_exact = 0.12 * min(opp_stack, pot)

        if self.opp_bid_lower_bounds:
            lb = sum(self.opp_bid_lower_bounds) / float(len(self.opp_bid_lower_bounds))
            pred = 0.72 * mean_exact + 0.28 * lb
        else:
            pred = mean_exact

        # Slightly conservative against overbidding opponents.
        pred *= 1.04
        return int(self._clip(pred, 0, opp_stack))

    def _is_overbidding_opponent(self, opp_stack: int, pot: int, pred_opp_bid: int) -> bool:
        if opp_stack <= 0:
            return False
        stack_ratio = pred_opp_bid / float(max(1, opp_stack))
        pot_ratio = pred_opp_bid / float(max(1, pot))
        fixed_samples = 0
        for x in self.opp_bid_exact_samples:
            if x in (398, 597, 1245):
                fixed_samples += 1
        fixed_rate = fixed_samples / float(max(1, len(self.opp_bid_exact_samples)))
        return (stack_ratio > 0.20 and pot_ratio > 1.6) or (fixed_rate > 0.28)

    def _is_micro_bidder(self, opp_stack: int, pot: int, pred_opp_bid: int) -> bool:
        if not self.opp_bid_exact_samples:
            return pred_opp_bid <= max(10, int(0.07 * pot))
        tiny = 0
        for x in self.opp_bid_exact_samples:
            if x <= 8:
                tiny += 1
        tiny_rate = tiny / float(max(1, len(self.opp_bid_exact_samples)))
        return tiny_rate > 0.48 and pred_opp_bid <= max(14, int(0.09 * pot))

    def _is_fixed_mid_bidder(self, opp_stack: int, pot: int, pred_opp_bid: int) -> bool:
        if not self.opp_bid_exact_samples:
            return False
        in_band = 0
        for x in self.opp_bid_exact_samples:
            if 8 <= x <= 12:
                in_band += 1
        band_rate = in_band / float(max(1, len(self.opp_bid_exact_samples)))
        return band_rate > 0.42 and 6 <= pred_opp_bid <= max(24, int(0.14 * pot))

    def _bankroll_factor(self, bankroll: int, round_num: int) -> float:
        # Normalize by expected single-stack scale.
        scale = 5000.0
        horizon = max(1.0, 1.0 - (round_num / 1200.0))
        return self._clip((bankroll / scale) * horizon, -1.0, 1.0)

    def _per_decision_time_budget(self, time_bank: float) -> float:
        # Keep large safety margin for match-total time.
        if time_bank < 2.0:
            return 0.020
        if time_bank < 5.0:
            return 0.032
        if time_bank < 10.0:
            return 0.045
        return 0.060

    def _rate(self, num: int, den: int, default: float) -> float:
        if den <= 0:
            return default
        return num / float(den)

    def _clip(self, x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x


if __name__ == '__main__':
    run_bot(Player(), parse_args())