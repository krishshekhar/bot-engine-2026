"""
Bot5 standalone: internal ensemble of CFR-like + exploitative policies.
"""
from collections import defaultdict, deque
import random
import time
import math

import eval7

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot
from pkbot.states import GameInfo, PokerState


RANK_ORDER = "23456789TJQKA"
RANK_TO_INT = {r: i + 2 for i, r in enumerate(RANK_ORDER)}
RANK_GRID = "AKQJT98765432"


class Player(BaseBot):
    def __init__(self) -> None:
        self.rng = random.Random(505)
        self.round_num = 0
        self.current_mode = 1

        # CFR-like memory
        self.regret_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.strategy_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        # Shared opponent model
        self.our_raise_opps = 0
        self.opp_fold_to_raise_hits = 0
        self.opp_post_bet_spots = 0
        self.opp_small_stab_hits = 0
        self.opp_huge_bet_hits = 0
        self.opp_pre_raise_spots = 0
        self.opp_small_pre_hits = 0
        self.opp_massive_pre_hits = 0
        self.opp_pre_rejam_spots = 0
        self.opp_pre_rejam_hits = 0

        self.opp_bid_exact = deque(maxlen=260)
        self.opp_bid_lb = deque(maxlen=260)

        self.hand_my_bid = None
        self.hand_auction_snapshot = None
        self.hand_auction_processed = False
        self.hand_i_raised = False
        self.hand_pre_raised = False

        self.street_action_count = {}
        self.equity_cache = {}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.round_num = game_info.round_num
        self.rng.seed((self.round_num * 3571) + (17 if current_state.is_bb else 11))
        self.street_action_count = {}
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
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        t0 = time.perf_counter()
        self.current_mode = self._mode(game_info.bankroll, game_info.round_num)
        self._update_models(current_state)
        self.street_action_count[current_state.street] = self.street_action_count.get(current_state.street, 0) + 1

        if game_info.time_bank < 0.8:
            return self._panic_action(current_state)

        a3 = self._policy3_action(game_info, current_state, t0)
        a4 = self._policy4_action(game_info, current_state, t0)

        if current_state.street == "auction":
            if self.hand_auction_snapshot is None:
                self.hand_auction_snapshot = (current_state.my_chips, current_state.opp_chips)
            act = self._merge_auction(current_state, a3, a4)
            if isinstance(act, ActionBid):
                self.hand_my_bid = int(act.amount)
            return act
        return self._merge_betting(current_state, a3, a4)

    # ---------- Internal policy #1 (CFR-like) ----------

    def _policy3_action(self, game_info: GameInfo, state: PokerState, t0: float):
        if state.street == "auction":
            eq = self._estimate_equity(state.my_hand, state.board, [], 40, t0, game_info.time_bank)
            pred = self._predict_opp_bid(state.opp_chips, state.pot)
            bucket = self._post_bucket100(eq, state.board)
            pred_bin = min(4, int((pred / float(max(1, state.opp_chips))) * 5))
            infoset = f"au|b{bucket}|p{pred_bin}"
            legal = [0, 1, 2, 3] if state.can_act(ActionBid) else [0]
            priors = self._auction_priors(eq, pred, state)
            utils = self._estimate_auction_utilities(state, eq, pred)
            choice = self._sample_regret_policy(infoset, legal, priors)
            self._update_regrets(infoset, legal, utils, choice)
            return ActionBid(self._auction_amount_from_action(state, choice, pred))

        if state.street == "pre-flop":
            bucket = self._preflop_bucket169(state.my_hand[0], state.my_hand[1])
            call_cost = max(0, state.cost_to_call)
            price_bin = min(4, int((call_cost / float(max(1, state.my_chips))) * 5))
            infoset = f"pf|b{bucket}|p{price_bin}|bb{1 if state.is_bb else 0}"
            legal = self._legal_action_indices(state, for_auction=False)
            eq = self._estimate_equity(state.my_hand, state.board, state.opp_revealed_cards, 36, t0, game_info.time_bank)
            utils = self._estimate_betting_utilities(state, eq)
            priors = self._preflop_priors(bucket)
            choice = self._sample_regret_policy(infoset, legal, priors)
            self._update_regrets(infoset, legal, utils, choice)
            return self._execute_abstract(state, choice, None)

        eq = self._estimate_equity(state.my_hand, state.board, state.opp_revealed_cards, 64, t0, game_info.time_bank)
        bucket = self._post_bucket100(eq, state.board)
        street_map = {"flop": 0, "turn": 1, "river": 2}
        call_cost = max(0, state.cost_to_call)
        rel = call_cost / float(max(1, state.pot))
        infoset = f"po|s{street_map.get(state.street, 0)}|b{bucket}|p{min(4,int(rel*5))}"
        legal = self._legal_action_indices(state, for_auction=False)
        utils = self._estimate_betting_utilities(state, eq)
        priors = self._postflop_priors(eq, state.board, rel)
        choice = self._sample_regret_policy(infoset, legal, priors)
        self._update_regrets(infoset, legal, utils, choice)
        return self._execute_abstract(state, choice, None)

    # ---------- Internal policy #2 (exploit/GTO hybrid) ----------

    def _policy4_action(self, game_info: GameInfo, state: PokerState, t0: float):
        if state.street == "auction":
            eq = self._estimate_equity(state.my_hand, state.board, [], 40, t0, game_info.time_bank)
            pot = max(1, state.pot)
            pred = self._predict_opp_bid(state.opp_chips, pot)
            if self._is_micro_bidder(pred, pot):
                bid = pred + (2 if eq < 0.68 else 1)
            elif self._is_high_mid_anchor_bidder(pred):
                if eq > 0.72 and self.rng.random() < 0.35:
                    bid = 0
                else:
                    bid = min(pred // 2, int(0.16 * pot))
            else:
                info = self._clip(0.22 - abs(eq - 0.5), 0.02, 0.22) * pot
                bid = int(0.54 * info + 0.46 * pred)
            n = self._neural_score(eq, 0.0, self._opp_fold_rate(), 0.0, self._board_wetness(state.board) / 3.0)
            bid = int(self._clip((0.80 + 0.35 * n) * bid + self.rng.randint(-6, 6), 0, state.my_chips))
            return ActionBid(bid)

        key = self._preflop_key(state.my_hand[0], state.my_hand[1])
        tier = self._preflop_tier(key)
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(max(1, pot + call_cost))
        small_pre = self._rate(self.opp_small_pre_hits, self.opp_pre_raise_spots, 0.25)
        massive_pre = self._rate(self.opp_massive_pre_hits, self.opp_pre_raise_spots, 0.08)
        rejam_rate = self._rate(self.opp_pre_rejam_hits, self.opp_pre_rejam_spots, 0.10)
        fold_rate = self._opp_fold_rate()

        if state.street == "pre-flop":
            if call_cost >= state.my_chips and state.can_act(ActionCall):
                # Guard against overcalling preflop jams with marginal strength.
                if tier >= 3:
                    return ActionCall()
                base = 0.24 if tier == 2 else 0.20 if tier == 1 else 0.15
                if massive_pre > 0.16:
                    base -= 0.02
                if self.current_mode == 2:
                    base += 0.02
                elif self.current_mode == 0:
                    base -= 0.02
                if pot_odds <= base:
                    return ActionCall()
                return ActionFold() if state.can_act(ActionFold) else ActionCall()

            if tier == 3:
                raise_p, call_p = 0.90, 0.08
            elif tier == 2:
                raise_p, call_p = 0.72, 0.20
            elif tier == 1:
                raise_p, call_p = 0.48, 0.34
            else:
                raise_p, call_p = 0.26, 0.30

            if fold_rate > 0.5:
                raise_p += 0.10
            if fold_rate < 0.24 and tier <= 1:
                raise_p -= 0.18
                call_p += 0.10
            if fold_rate < 0.18 and tier == 0:
                raise_p -= 0.14
                call_p += 0.06
            if small_pre > 0.45 and call_cost <= 100 and tier <= 1:
                raise_p -= 0.12
                call_p += 0.10
            if massive_pre > 0.16 and call_cost > 0.22 * state.my_chips and tier <= 1:
                raise_p *= 0.65
                call_p *= 0.70
            if massive_pre > 0.30 and tier <= 1:
                raise_p *= 0.45
                call_p *= 0.78
            if call_cost == 0 and massive_pre > 0.30 and tier <= 1:
                raise_p = min(raise_p, 0.16)
            # Opponent springing back-jams over our raises: avoid weak bloated 3-bets.
            if rejam_rate > 0.22 and tier <= 1:
                raise_p *= 0.55
                call_p *= 1.12
            if rejam_rate > 0.30 and call_cost <= 100 and tier <= 1:
                raise_p = min(raise_p, 0.14 if tier == 0 else 0.30)

            score = self._neural_score(self._tier_to_eq_proxy(tier), pot_odds, fold_rate, call_cost / float(max(1, state.my_chips)), 0.0)
            raise_p += 0.12 * (score - 0.5)
            call_p -= 0.08 * (score - 0.5)
            raise_p = self._clip(raise_p, 0.0, 1.0)
            call_p = self._clip(call_p, 0.0, 0.95)

            r = self.rng.random()
            if state.can_act(ActionRaise) and r < raise_p:
                return self._safe_raise(state, self._preflop_raise_amount(state, tier, small_pre, rejam_rate))
            if state.can_act(ActionCall) and r < raise_p + call_p:
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return ActionFold() if state.can_act(ActionFold) else self._fallback(state)

        # postflop
        eq = self._estimate_equity(state.my_hand, state.board, state.opp_revealed_cards, 64, t0, game_info.time_bank)
        wet = self._board_wetness(state.board)
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(max(1, pot + call_cost))
        rel_price = call_cost / float(max(1, pot))
        score = self._neural_score(eq, pot_odds, fold_rate, rel_price, wet / 3.0)

        if call_cost > 0:
            raise_th = 0.60 + 0.05 * max(0.0, rel_price - 0.5)
            call_th = pot_odds + 0.03 + (0.04 if rel_price > 0.85 else 0.0)
            if self._rate(self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18) > 0.22:
                call_th += 0.04
            if state.can_act(ActionRaise) and eq > raise_th and score > 0.48:
                return self._safe_raise(state, self._post_raise_amount(state, eq, rel_price, wet))
            if state.can_act(ActionCall) and eq >= call_th:
                return ActionCall()
            return ActionFold() if state.can_act(ActionFold) else self._fallback(state)
        if state.can_act(ActionRaise):
            if eq > (0.55 if state.street == "flop" else 0.58) or (fold_rate > 0.44 and score > 0.54):
                return self._safe_raise(state, self._post_raise_amount(state, eq, 0.0, wet))
        if state.can_act(ActionCheck):
            return ActionCheck()
        return self._fallback(state)

    # ---------- Ensemble merge ----------

    def _merge_auction(self, state: PokerState, a3, a4):
        if not state.can_act(ActionBid):
            return self._fallback(state)
        b3 = a3.amount if isinstance(a3, ActionBid) else 0
        b4 = a4.amount if isinstance(a4, ActionBid) else 0
        pred = self._predict_opp_bid(state.opp_chips, state.pot)
        if state.pot < 260 or state.my_chips < 1000:
            bid = int(0.28 * b4 + 0.72 * b3)
        elif state.pot > 500:
            bid = int(0.70 * b4 + 0.30 * b3)
        else:
            bid = int(0.52 * b4 + 0.48 * b3)
        if (b3 == 0 or b4 == 0) and state.pot < 220:
            bid = min(bid, max(b3, b4))
        # High-bidder tax: make expensive info purchases cost more.
        if len(self.opp_bid_exact) >= 8 and pred >= 120 and state.my_chips > 200:
            tax_floor = int(self._clip(max(12, 0.25 * pred), 0, state.my_chips))
            bid = max(bid, tax_floor)
            if pred >= 260 and state.pot < 420:
                contest = int(self._clip(0.42 * pred, 0, state.my_chips))
                bid = max(bid, contest)
        return ActionBid(int(self._clip(bid, 0, state.my_chips)))

    def _merge_betting(self, state: PokerState, a3, a4):
        idx3, r3 = self._to_abstract(state, a3)
        idx4, r4 = self._to_abstract(state, a4)
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        spr = state.my_chips / float(max(1, pot))
        huge_rate = self._rate(self.opp_huge_bet_hits, self.opp_post_bet_spots, 0.18)

        if call_cost > 0:
            rel = call_cost / float(pot)
            rejam_rate = self._rate(self.opp_pre_rejam_hits, self.opp_pre_rejam_spots, 0.10)
            # Hard anti-spew filter for massive pressure spots.
            if rel > 0.95 and (idx3 >= 2 or idx4 >= 2):
                if state.street == "pre-flop":
                    key = self._preflop_key(state.my_hand[0], state.my_hand[1])
                    tier = self._preflop_tier(key)
                    chosen = 1 if tier >= 2 else 0
                else:
                    chosen = 1 if huge_rate < 0.26 else 0
            elif rel > 0.78 and huge_rate > 0.24 and max(idx3, idx4) >= 2:
                chosen = 1
            else:
                if rel > 0.70 or spr < 1.2:
                    chosen = idx4
                    if rel > 0.95 and idx3 <= 1 and idx4 >= 2:
                        chosen = 1
                elif rel < 0.25 and pot < 260:
                    chosen = idx3
                else:
                    if idx3 == idx4:
                        chosen = idx3
                    elif idx3 <= 1 and idx4 >= 2:
                        chosen = 2 if rel < 0.55 else 1
                    elif idx4 <= 1 and idx3 >= 2:
                        chosen = 2 if rel < 0.45 else 1
                    else:
                        chosen = max(idx3, idx4)

            if self.current_mode == 0 and rel > 0.55 and chosen >= 2:
                chosen = 1
            if state.street == "pre-flop" and rejam_rate > 0.24 and chosen >= 2:
                key = self._preflop_key(state.my_hand[0], state.my_hand[1])
                tier = self._preflop_tier(key)
                if tier <= 1:
                    chosen = 1 if rel < 0.45 else 0
        else:
            if pot < 240:
                chosen = 1 if idx3 == 0 else idx3
            else:
                chosen = 1 if (idx3 == 0 and idx4 == 0) else max(idx3, idx4)

            if self.current_mode == 2 and chosen == 1 and max(idx3, idx4) >= 2 and pot > 220:
                chosen = 2

        pref = None
        if r3 is not None and r4 is not None:
            if call_cost > 0 and (call_cost / float(pot) > 0.60 or spr < 1.4):
                pref = int(0.30 * r3 + 0.70 * r4)
            else:
                pref = int((r3 + r4) / 2)
        elif r3 is not None:
            pref = r3
        elif r4 is not None:
            pref = r4
        return self._execute_abstract(state, chosen, pref)

    # ---------- Common action abstraction ----------

    def _to_abstract(self, state: PokerState, action):
        if isinstance(action, ActionRaise):
            lo, hi = state.raise_bounds
            if action.amount >= hi - max(2, int(0.08 * max(1, hi - lo))):
                return 3, int(action.amount)
            return 2, int(action.amount)
        if isinstance(action, (ActionCall, ActionCheck)):
            return 1, None
        return 0, None

    def _execute_abstract(self, state: PokerState, idx: int, preferred_raise: int | None):
        if idx == 0:
            if state.can_act(ActionFold):
                return ActionFold()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return self._fallback(state)
        if idx == 1:
            if state.can_act(ActionCall):
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return self._fallback(state)
        if state.can_act(ActionRaise):
            lo, hi = state.raise_bounds
            if idx == 3:
                amt = hi
            else:
                if preferred_raise is None:
                    if state.cost_to_call > 0:
                        amt = state.opp_wager + state.pot
                    else:
                        amt = state.my_wager + int(0.66 * state.pot)
                else:
                    amt = preferred_raise
            return self._safe_raise(state, int(self._clip(amt, lo, hi)))
        return self._fallback(state)

    # ---------- Shared helpers ----------

    def _sample_regret_policy(self, infoset: str, legal_actions: list[int], priors: list[float]) -> int:
        regrets = self.regret_sum[infoset]
        positive = [max(0.0, regrets[i]) if i in legal_actions else 0.0 for i in range(4)]
        total_pos = sum(positive)
        probs = [0.0, 0.0, 0.0, 0.0]
        if total_pos > 1e-9:
            for i in legal_actions:
                probs[i] = positive[i] / total_pos
        else:
            total_prior = sum(priors[i] for i in legal_actions)
            for i in legal_actions:
                probs[i] = (priors[i] / total_prior) if total_prior > 1e-9 else 1.0 / float(len(legal_actions))
        for i in legal_actions:
            self.strategy_sum[infoset][i] += probs[i]
        r = self.rng.random()
        acc = 0.0
        for i in legal_actions:
            acc += probs[i]
            if r <= acc:
                return i
        return legal_actions[-1]

    def _update_regrets(self, infoset: str, legal_actions: list[int], utilities: list[float], chosen: int) -> None:
        chosen_u = utilities[chosen]
        for i in legal_actions:
            self.regret_sum[infoset][i] += utilities[i] - chosen_u

    def _estimate_betting_utilities(self, state: PokerState, eq: float) -> list[float]:
        pot = max(1, state.pot)
        call_cost = max(0, state.cost_to_call)
        _, max_r = state.raise_bounds
        fold_rate = self._opp_fold_rate()
        rel_price = call_cost / float(max(1, pot))
        spr = state.my_chips / float(max(1, pot))
        u = [-1e9, -1e9, -1e9, -1e9]
        if state.can_act(ActionCheck):
            u[0] = 0.05 * pot
        elif state.can_act(ActionFold):
            u[0] = 0.0
        if state.can_act(ActionCall):
            u[1] = eq * (pot + call_cost) - (1.0 - eq) * call_cost
        elif state.can_act(ActionCheck):
            u[1] = 0.04 * pot
        if state.can_act(ActionRaise):
            mid_amt = self._medium_raise_amount(state)
            mid_risk = max(0, mid_amt - state.my_wager)
            jam_risk = max(0, max_r - state.my_wager)
            fold_mid = self._clip(fold_rate + 0.04, 0.08, 0.76)
            fold_jam = self._clip(fold_rate - 0.10, 0.03, 0.62)
            jam_penalty = 0.0
            if rel_price > 0.65 and eq < 0.60:
                jam_penalty += (0.60 - eq) * 0.42 * jam_risk
            if spr > 1.7 and eq < 0.57:
                jam_penalty += (0.57 - eq) * 0.26 * jam_risk
            if self.current_mode == 0 and eq < 0.62:
                jam_penalty += (0.62 - eq) * 0.18 * jam_risk
            if fold_rate < 0.24 and eq < 0.62:
                jam_penalty += (0.62 - eq) * 0.24 * jam_risk
            u[2] = fold_mid * pot + (1.0 - fold_mid) * (eq * (pot + 2 * mid_risk) - (1.0 - eq) * mid_risk)
            u[3] = fold_jam * pot + (1.0 - fold_jam) * (eq * (pot + 2 * jam_risk) - (1.0 - eq) * jam_risk) - jam_penalty
        return u

    def _estimate_auction_utilities(self, state: PokerState, eq: float, pred: int) -> list[float]:
        pot = max(1, state.pot)
        cap = state.my_chips
        a0 = 0
        a1 = max(1, min(cap, pred))
        a2 = max(2, min(cap, int(0.45 * pot)))
        a3 = min(cap, max(a2, int(0.20 * min(state.my_chips, state.opp_chips))))
        amounts = [a0, a1, a2, a3]
        info_value = self._clip(0.24 - abs(eq - 0.5), 0.02, 0.24) * pot
        out = []
        for a in amounts:
            win_prob = self._clip(0.5 + (a - pred) / float(max(1, 2 * max(1, pred))), 0.05, 0.95)
            out.append(win_prob * info_value - 0.72 * min(a, cap))
        return out

    def _preflop_priors(self, bucket: int) -> list[float]:
        strength = 1.0 - (bucket / 168.0)
        p0 = self._clip(0.20 - 0.18 * strength, 0.02, 0.30)
        p1 = self._clip(0.34 - 0.12 * strength, 0.15, 0.40)
        p2 = self._clip(0.32 + 0.10 * strength, 0.22, 0.46)
        p3 = self._clip(0.14 + 0.12 * strength, 0.06, 0.30)
        return [p0, p1, p2, p3]

    def _postflop_priors(self, eq: float, board: list[str], rel_price: float) -> list[float]:
        wet = self._board_wetness(board)
        p0 = self._clip(0.24 - 0.20 * eq + 0.08 * rel_price, 0.03, 0.38)
        p1 = self._clip(0.30 - 0.10 * eq + 0.10 * wet, 0.14, 0.42)
        p2 = self._clip(0.34 + 0.14 * eq - 0.04 * wet, 0.18, 0.50)
        p3 = self._clip(0.10 + 0.14 * eq + 0.03 * max(0.0, rel_price - 0.4), 0.03, 0.32)
        if self.current_mode == 0:
            p0 += 0.04
            p1 += 0.03
            p3 -= 0.06
        elif self.current_mode == 2:
            p2 += 0.03
            p3 += 0.02
        return [p0, p1, p2, p3]

    def _auction_priors(self, eq: float, pred: int, state: PokerState) -> list[float]:
        low_stack = state.my_chips < 0.35 * max(1, state.opp_chips)
        p0 = 0.18 if eq < 0.75 else 0.30
        p1 = 0.42
        p2 = 0.30 if not low_stack else 0.20
        p3 = 0.10 if not low_stack else 0.08
        if pred > 120:
            p1 += 0.12
            p2 += 0.08
            p0 -= 0.06
            p3 -= 0.04
        return [p0, p1, p2, p3]

    def _auction_amount_from_action(self, state: PokerState, a: int, pred: int) -> int:
        cap = state.my_chips
        pot = max(1, state.pot)
        if a == 0:
            return 0
        if a == 1:
            return int(self._clip(min(pred + 1, int(0.45 * pot) + 4), 0, cap))
        if a == 2:
            return int(self._clip(min(int(0.45 * pot), pred + 10), 0, cap))
        return int(self._clip(int(0.20 * min(state.my_chips, state.opp_chips)), 0, cap))

    def _legal_action_indices(self, state: PokerState, for_auction: bool) -> list[int]:
        if for_auction:
            return [0, 1, 2, 3] if state.can_act(ActionBid) else [0]
        legal = []
        if state.can_act(ActionFold) or state.can_act(ActionCheck):
            legal.append(0)
        if state.can_act(ActionCall) or state.can_act(ActionCheck):
            legal.append(1)
        if state.can_act(ActionRaise):
            legal.extend([2, 3])
        return sorted(set(legal or [0]))

    def _medium_raise_amount(self, state: PokerState) -> int:
        lo, hi = state.raise_bounds
        pot = max(1, state.pot)
        target = state.opp_wager + int(1.00 * pot) if state.cost_to_call > 0 else state.my_wager + int(0.66 * pot)
        return int(self._clip(target, lo, hi))

    def _preflop_bucket169(self, c1: str, c2: str) -> int:
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        i1 = RANK_GRID.index(r1)
        i2 = RANK_GRID.index(r2)
        if r1 == r2:
            i, j = i1, i2
        elif s1 == s2:
            i, j = min(i1, i2), max(i1, i2)
        else:
            i, j = max(i1, i2), min(i1, i2)
        return i * 13 + j

    def _post_bucket100(self, eq: float, board: list[str]) -> int:
        adj = self._clip(eq + 0.015 * self._board_wetness(board), 0.0, 1.0)
        return int(self._clip(int(adj * 99), 0, 99))

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
        frac = 0.92 if tier >= 3 else 0.80 if tier == 2 else 0.65 if tier == 1 else 0.50
        if small_pre > 0.45 and tier <= 1:
            frac -= 0.14
        if rejam_rate > 0.22 and tier <= 1:
            frac -= 0.20 if tier == 0 else 0.12
        if rejam_rate > 0.30 and tier == 0:
            frac = min(frac, 0.18)
        frac += self.rng.uniform(-0.05, 0.05)
        return int(self._clip(lo + int(span * self._clip(frac, 0.06, 0.97)), lo, hi))

    def _post_raise_amount(self, state: PokerState, eq: float, rel_price: float, wet: int) -> int:
        lo, hi = state.raise_bounds
        span = max(0, hi - lo)
        frac = self._clip(0.58 + 0.30 * (eq - 0.50) + 0.10 * max(0.0, rel_price - 0.4) - 0.04 * wet, 0.18, 0.96)
        if self.current_mode == 0:
            frac -= 0.10
        elif self.current_mode == 2:
            frac += 0.06
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

    def _update_models(self, state: PokerState) -> None:
        if state.street == "pre-flop" and state.cost_to_call > 0:
            self.opp_pre_raise_spots += 1
            if state.opp_wager <= 120 or state.cost_to_call <= 80:
                self.opp_small_pre_hits += 1
            if state.opp_wager >= 900 or state.cost_to_call >= 850:
                self.opp_massive_pre_hits += 1
            if self.hand_pre_raised:
                self.opp_pre_rejam_spots += 1
                if state.opp_wager >= 1500 or state.cost_to_call >= max(500, int(0.25 * max(1, state.my_chips))):
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
                self.opp_bid_exact.append(d_my)
            elif d_opp > 0 and d_my == 0:
                self.opp_bid_lb.append(max(1, self.hand_my_bid + 1))
            elif d_my > 0 and d_opp > 0:
                self.opp_bid_exact.append(d_opp)
            self.hand_auction_processed = True

    def _predict_opp_bid(self, opp_stack: int, pot: int) -> int:
        if self.opp_bid_exact:
            ex = sum(self.opp_bid_exact) / float(len(self.opp_bid_exact))
        else:
            ex = 0.10 * min(opp_stack, pot)
        if self.opp_bid_lb:
            lb = sum(self.opp_bid_lb) / float(len(self.opp_bid_lb))
            pred = 0.70 * ex + 0.30 * lb
        else:
            pred = ex
        return int(self._clip(pred, 0, opp_stack))

    def _is_micro_bidder(self, pred: int, pot: int) -> bool:
        if not self.opp_bid_exact:
            return pred <= max(10, int(0.08 * pot))
        return sum(1 for x in self.opp_bid_exact if x <= 8) / float(len(self.opp_bid_exact)) > 0.45

    def _is_high_mid_anchor_bidder(self, pred: int) -> bool:
        if len(self.opp_bid_exact) < 12:
            return False
        band = sum(1 for x in self.opp_bid_exact if 220 <= x <= 280)
        return band / float(len(self.opp_bid_exact)) > 0.30 and pred >= 170

    def _opp_fold_rate(self) -> float:
        if self.our_raise_opps < 8:
            return 0.34
        return self.opp_fold_to_raise_hits / float(max(1, self.our_raise_opps))

    def _safe_raise(self, state: PokerState, amount: int):
        if state.can_act(ActionRaise):
            lo, hi = state.raise_bounds
            amt = int(self._clip(amount, lo, hi))
            self.hand_i_raised = True
            if state.street == "pre-flop":
                self.hand_pre_raised = True
            return ActionRaise(amt)
        return self._fallback(state)

    def _fallback(self, state: PokerState):
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        if state.can_act(ActionBid):
            return ActionBid(0)
        return ActionFold()

    def _panic_action(self, state: PokerState):
        if state.street == "auction":
            return ActionBid(int(self._clip(min(4, state.my_chips), 0, state.my_chips)))
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        return ActionFold()

    def _estimate_equity(
        self,
        my_hand: list[str],
        board: list[str],
        opp_revealed: list[str],
        iters: int,
        t0: float,
        time_bank: float,
    ) -> float:
        key = (tuple(sorted(my_hand)), tuple(board), tuple(sorted(opp_revealed)), len(board), iters // 20)
        if key in self.equity_cache:
            return self.equity_cache[key]
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
        wins = 0.0
        n = 0
        budget = self._decision_budget(time_bank)
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
        if len(self.equity_cache) > 6000:
            self.equity_cache.clear()
        self.equity_cache[key] = eq
        return eq

    @staticmethod
    def _decision_budget(time_bank: float) -> float:
        if time_bank < 2.0:
            return 0.010
        if time_bank < 5.0:
            return 0.016
        if time_bank < 10.0:
            return 0.025
        return 0.036

    def _neural_score(self, eq_proxy: float, pot_odds: float, fold_rate: float, pressure: float, wetness: float) -> float:
        h1 = math.tanh(1.4 * eq_proxy - 0.9 * pot_odds + 0.7 * fold_rate - 1.1 * pressure - 0.3 * wetness + 0.15)
        h2 = math.tanh(1.0 * eq_proxy + 0.6 * pot_odds + 0.4 * fold_rate - 0.7 * pressure - 0.2 * wetness - 0.10)
        h3 = math.tanh(0.8 * eq_proxy - 0.4 * pot_odds + 0.9 * fold_rate - 0.6 * pressure - 0.1 * wetness + 0.05)
        z = 1.2 * h1 + 0.8 * h2 + 0.9 * h3 - 0.15
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _mode(bankroll: int, round_num: int) -> int:
        # 0 survival, 1 balanced, 2 chase/extraction.
        if round_num > 850 and bankroll > 90000:
            return 0
        if bankroll < -50000:
            return 2
        return 1

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
