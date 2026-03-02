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

from botharshu_v4 import RANK_TO_INT
from botharshu_v4_2 import Player as V42Player  # type: ignore


class Player(V42Player):
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

