"""
Bot3: CFR-inspired abstraction bot for Sneak Peek Hold'em.

This is not full-scale offline CFR training; it applies CFR-style ideas online:
- hand-state abstraction (preflop 169 buckets, postflop 100 buckets),
- discretized betting abstraction (check/fold, call/check, medium, all-in),
- imperfect recall (current street bucket only),
- regret-matching-lite updates from fast counterfactual utility estimates.
"""
from collections import defaultdict, deque
import random
import time

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
        self.rng = random.Random(303)

        self.regret_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.strategy_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        self.our_raise_opps = 0
        self.opp_fold_to_raise_hits = 0

        self.opp_bid_exact = deque(maxlen=250)
        self.opp_bid_lb = deque(maxlen=250)
        self.hand_my_bid = None
        self.hand_auction_snapshot = None
        self.hand_auction_processed = False
        self.hand_i_raised = False

        self.street_action_count = {}
        self.equity_cache = {}

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        self.rng.seed((game_info.round_num * 1009) + (17 if current_state.is_bb else 11))
        self.street_action_count = {}
        self.hand_my_bid = None
        self.hand_auction_snapshot = None
        self.hand_auction_processed = False
        self.hand_i_raised = False

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        if self.hand_i_raised:
            self.our_raise_opps += 1
            if len(current_state.opp_revealed_cards) < 2 and current_state.payoff > 0:
                self.opp_fold_to_raise_hits += 1

    def get_move(
        self, game_info: GameInfo, current_state: PokerState
    ) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        t0 = time.perf_counter()
        self._update_models(current_state)

        if game_info.time_bank < 0.8:
            return self._panic_action(current_state)

        street = current_state.street
        self.street_action_count[street] = self.street_action_count.get(street, 0) + 1

        if street == "auction":
            return self._play_auction(game_info, current_state, t0)
        if street == "pre-flop":
            return self._play_preflop(current_state, t0, game_info.time_bank)
        return self._play_postflop(current_state, t0, game_info.time_bank)

    # ---------- Core action logic ----------

    def _play_preflop(self, state: PokerState, t0: float, time_bank: float):
        bucket = self._preflop_bucket169(state.my_hand[0], state.my_hand[1])
        call_cost = max(0, state.cost_to_call)
        price_bin = min(4, int((call_cost / float(max(1, state.my_chips))) * 5))
        action_cap = min(2, self.street_action_count.get("pre-flop", 1))

        infoset = f"pf|b{bucket}|p{price_bin}|a{action_cap}|bb{1 if state.is_bb else 0}"
        legal = self._legal_action_indices(state, for_auction=False)

        # Cap to two aggressive decisions per street in abstraction spirit.
        if action_cap > 2:
            legal = [i for i in legal if i in (0, 1)] or legal

        eq = self._estimate_equity(state.my_hand, state.board, state.opp_revealed_cards, 36, t0, time_bank)
        utils = self._estimate_betting_utilities(state, eq)
        priors = self._preflop_priors(bucket)
        choice = self._sample_regret_policy(infoset, legal, priors)
        self._update_regrets(infoset, legal, utils, choice)

        return self._execute_betting_action(state, choice)

    def _play_postflop(self, state: PokerState, t0: float, time_bank: float):
        eq = self._estimate_equity(state.my_hand, state.board, state.opp_revealed_cards, 64, t0, time_bank)
        bucket = self._post_bucket100(eq, state.board)
        street_map = {"flop": 0, "turn": 1, "river": 2}
        street_id = street_map.get(state.street, 0)
        call_cost = max(0, state.cost_to_call)
        rel = call_cost / float(max(1, state.pot))
        pressure_bin = min(4, int(rel * 5))
        action_cap = min(2, self.street_action_count.get(state.street, 1))

        infoset = f"po|s{street_id}|b{bucket}|p{pressure_bin}|a{action_cap}"
        legal = self._legal_action_indices(state, for_auction=False)
        if action_cap > 2:
            legal = [i for i in legal if i in (0, 1)] or legal

        utils = self._estimate_betting_utilities(state, eq)
        priors = self._postflop_priors(eq, state.board, rel)
        choice = self._sample_regret_policy(infoset, legal, priors)
        self._update_regrets(infoset, legal, utils, choice)
        if rel >= 0.80 and eq < (call_cost / float(max(1, state.pot + call_cost)) + 0.07):
            if state.can_act(ActionFold):
                return ActionFold()
            if state.can_act(ActionCall):
                return ActionCall()

        return self._execute_betting_action(state, choice)

    def _play_auction(self, game_info: GameInfo, state: PokerState, t0: float):
        self.hand_auction_snapshot = (state.my_chips, state.opp_chips)
        eq = self._estimate_equity(state.my_hand, state.board, [], 38, t0, game_info.time_bank)
        bucket = self._post_bucket100(eq, state.board)
        pred = self._predict_opp_bid(state.opp_chips, state.pot)
        pred_bin = min(4, int((pred / float(max(1, state.opp_chips))) * 5))
        infoset = f"au|b{bucket}|p{pred_bin}"

        legal = self._legal_action_indices(state, for_auction=True)
        priors = self._auction_priors(eq, pred, state)
        utils = self._estimate_auction_utilities(state, eq, pred)
        choice = self._sample_regret_policy(infoset, legal, priors)
        self._update_regrets(infoset, legal, utils, choice)

        bid = self._auction_amount_from_action(state, choice, pred)
        self.hand_my_bid = bid
        return self._safe_bid(state, bid)

    # ---------- CFR-like policy ----------

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
            if total_prior <= 1e-9:
                for i in legal_actions:
                    probs[i] = 1.0 / float(len(legal_actions))
            else:
                for i in legal_actions:
                    probs[i] = priors[i] / total_prior

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

    # ---------- Utility estimators ----------

    def _estimate_betting_utilities(self, state: PokerState, eq: float) -> list[float]:
        pot = max(1, state.pot)
        call_cost = max(0, state.cost_to_call)
        _, max_r = state.raise_bounds
        fold_rate = self._opp_fold_to_raise_rate()
        rel_price = call_cost / float(max(1, pot))

        # 0: check/fold, 1: call/check, 2: medium, 3: all-in.
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
            fold_jam = self._clip(fold_rate - 0.08, 0.04, 0.66)
            jam_penalty = 0.0
            if rel_price > 0.65 and eq < 0.58:
                jam_penalty = (0.58 - eq) * 0.35 * jam_risk

            u[2] = (
                fold_mid * pot
                + (1.0 - fold_mid) * (eq * (pot + 2 * mid_risk) - (1.0 - eq) * mid_risk)
            )
            u[3] = (
                fold_jam * pot
                + (1.0 - fold_jam) * (eq * (pot + 2 * jam_risk) - (1.0 - eq) * jam_risk)
            ) - jam_penalty

        return u

    def _estimate_auction_utilities(self, state: PokerState, eq: float, pred: int) -> list[float]:
        pot = max(1, state.pot)
        my_cap = state.my_chips
        a0 = 0
        a1 = max(1, min(my_cap, pred))
        a2 = max(2, min(my_cap, int(0.66 * pot)))
        a3 = min(my_cap, max(a2, int(0.28 * min(state.my_chips, state.opp_chips))))
        amounts = [a0, a1, a2, a3]

        # Win prob proxy from relation to predicted opponent bid.
        utils = []
        info_value = self._clip(0.24 - abs(eq - 0.5), 0.02, 0.24) * pot
        for a in amounts:
            win_prob = self._clip(0.5 + (a - pred) / float(max(1, 2 * max(1, pred))), 0.05, 0.95)
            pay = min(a, my_cap)
            utils.append(win_prob * info_value - 0.72 * pay)
        return utils

    # ---------- Priors ----------

    def _preflop_priors(self, bucket: int) -> list[float]:
        strength = 1.0 - (bucket / 168.0)
        # [fold/check, call/check, medium, allin]
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
        p3 = self._clip(0.12 + 0.18 * eq + 0.04 * max(0.0, rel_price - 0.4), 0.05, 0.40)
        return [p0, p1, p2, p3]

    def _auction_priors(self, eq: float, pred: int, state: PokerState) -> list[float]:
        low_stack = state.my_chips < 0.35 * max(1, state.opp_chips)
        p0 = 0.18 if eq < 0.75 else 0.30
        p1 = 0.42
        p2 = 0.30 if not low_stack else 0.20
        p3 = 0.10 if not low_stack else 0.08
        # small push if predicted opponent bid is high
        if pred > 120:
            p1 += 0.12
            p3 -= 0.04
        return [p0, p1, p2, p3]

    # ---------- Action execution ----------

    def _execute_betting_action(self, state: PokerState, abstract_action: int):
        if abstract_action == 0:
            if state.can_act(ActionCheck):
                return ActionCheck()
            if state.can_act(ActionFold):
                return ActionFold()
            return self._fallback_action(state)

        if abstract_action == 1:
            if state.can_act(ActionCall):
                return ActionCall()
            if state.can_act(ActionCheck):
                return ActionCheck()
            return self._fallback_action(state)

        if abstract_action == 2:
            if state.can_act(ActionRaise):
                amt = self._medium_raise_amount(state)
                return self._safe_raise(state, amt)
            return self._fallback_action(state)

        if state.can_act(ActionRaise):
            _, max_r = state.raise_bounds
            return self._safe_raise(state, max_r)
        return self._fallback_action(state)

    def _auction_amount_from_action(self, state: PokerState, a: int, pred: int) -> int:
        my_cap = state.my_chips
        pot = max(1, state.pot)
        if a == 0:
            return 0
        if a == 1:
            return int(self._clip(min(pred + 1, int(0.45 * pot) + 4), 0, my_cap))
        if a == 2:
            return int(self._clip(min(int(0.45 * pot), pred + 10), 0, my_cap))
        return int(self._clip(int(0.20 * min(state.my_chips, state.opp_chips)), 0, my_cap))

    # ---------- Safety / fallback ----------

    def _safe_raise(self, state: PokerState, amount: int):
        if state.can_act(ActionRaise):
            lo, hi = state.raise_bounds
            amt = int(self._clip(amount, lo, hi))
            self.hand_i_raised = True
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
            return self._safe_bid(state, min(5, state.my_chips))
        if state.can_act(ActionCheck):
            return ActionCheck()
        if state.can_act(ActionCall):
            return ActionCall()
        return ActionFold()

    # ---------- Models / buckets ----------

    def _update_models(self, state: PokerState) -> None:
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

    def _preflop_bucket169(self, c1: str, c2: str) -> int:
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        i1 = RANK_GRID.index(r1)
        i2 = RANK_GRID.index(r2)
        if r1 == r2:
            i, j = i1, i2
        elif s1 == s2:
            i, j = min(i1, i2), max(i1, i2)  # suited upper triangle
        else:
            i, j = max(i1, i2), min(i1, i2)  # offsuit lower triangle
        return i * 13 + j

    def _post_bucket100(self, eq: float, board: list[str]) -> int:
        wet = self._board_wetness(board)
        adj = self._clip(eq + 0.015 * wet, 0.0, 1.0)
        return int(self._clip(int(adj * 99), 0, 99))

    def _board_wetness(self, board: list[str]) -> int:
        if not board:
            return 0
        ranks = sorted([RANK_TO_INT[c[0]] for c in board], reverse=True)
        suits = [c[1] for c in board]
        max_suit = max(suits.count("s"), suits.count("h"), suits.count("d"), suits.count("c"))
        conn = 0
        for i in range(len(ranks) - 1):
            if abs(ranks[i] - ranks[i + 1]) <= 2:
                conn += 1
        paired = len(set(ranks)) < len(ranks)
        w = 0
        if max_suit >= 3:
            w += 1
        if conn >= 2:
            w += 1
        if paired:
            w += 1
        return w

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
        if not legal:
            legal = [0]
        return sorted(set(legal))

    def _medium_raise_amount(self, state: PokerState) -> int:
        lo, hi = state.raise_bounds
        pot = max(1, state.pot)
        if state.cost_to_call <= 0:
            target = state.my_wager + int(0.66 * pot)
        else:
            target = state.opp_wager + int(1.00 * pot)
        return int(self._clip(target, lo, hi))

    def _predict_opp_bid(self, opp_stack: int, pot: int) -> int:
        if self.opp_bid_exact:
            ex = sum(self.opp_bid_exact) / float(len(self.opp_bid_exact))
        else:
            ex = 0.10 * min(opp_stack, pot)
        if self.opp_bid_lb:
            lb = sum(self.opp_bid_lb) / float(len(self.opp_bid_lb))
            pred = 0.72 * ex + 0.28 * lb
        else:
            pred = ex
        return int(self._clip(pred, 0, opp_stack))

    def _opp_fold_to_raise_rate(self) -> float:
        if self.our_raise_opps < 8:
            return 0.30
        return self.opp_fold_to_raise_hits / float(max(1, self.our_raise_opps))

    # ---------- Equity ----------

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

        max_runtime = self._per_decision_budget(time_bank)
        wins = 0.0
        n = 0
        for _ in range(max(16, iters)):
            if time.perf_counter() - t0 > max_runtime:
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
            my_score = eval7.evaluate(my_cards + full_board)
            opp_score = eval7.evaluate(opp + full_board)
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

    @staticmethod
    def _per_decision_budget(time_bank: float) -> float:
        if time_bank < 2.0:
            return 0.010
        if time_bank < 5.0:
            return 0.016
        if time_bank < 10.0:
            return 0.025
        return 0.036

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    @staticmethod
    def _rate(num: int, den: int, default: float) -> float:
        if den <= 0:
            return default
        return num / float(den)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
