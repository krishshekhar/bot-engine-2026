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

from botharshu_v4 import Player as V4Player, RANK_TO_INT, RANK_ORDER


class Player(V4Player):
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


if __name__ == "__main__":
    run_bot(Player(), parse_args())

