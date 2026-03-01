"""
Bot4 Ver3: Bot4 variant with anti-sticky postflop control.

Goal:
- Keep Bot4's strong preflop + auction logic.
- Reduce postflop over-barreling against call-heavy opponents.
"""
from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.runner import parse_args, run_bot

import bot4


class Player(bot4.Player):
    def _play_preflop(self, game_info, state, t0):
        key = self._preflop_key(state.my_hand[0], state.my_hand[1])
        tier = self._preflop_tier(key)
        call_cost = max(0, state.cost_to_call)
        pot = max(1, state.pot)
        pot_odds = call_cost / float(max(1, pot + call_cost))
        rejam_rate = self._rate(self.opp_pre_rejam_hits, self.opp_pre_rejam_spots, 0.10)

        # Hard brake: when opponents often re-jam after our raise, fold more weak continues.
        if self.hand_pre_raised and call_cost > 0 and rejam_rate > 0.14:
            price = call_cost / float(max(1, state.my_chips))
            if tier <= 1 and (price > 0.06 or pot_odds > 0.18):
                return ActionFold() if state.can_act(ActionFold) else ActionCall()
            if tier == 2 and (price > 0.16 and pot_odds > 0.23):
                return ActionFold() if state.can_act(ActionFold) else ActionCall()

        return super()._play_preflop(game_info, state, t0)

    def _play_postflop(self, game_info, state, t0):
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
        sticky = fold_rate < 0.30
        medium_pot = 650 <= pot <= 3200

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

            # Tighten aggression and calling thresholds against sticky callers.
            if sticky:
                raise_th += 0.06
                call_th += 0.05
                if rel_price > 0.55:
                    call_th += 0.03
            # Medium-pot commitment control: avoid paying off too light in bloated pots.
            if medium_pot:
                raise_th += 0.03
                call_th += 0.03
                if state.street in ("turn", "river"):
                    call_th += 0.03
                if rel_price > 0.35:
                    call_th += 0.02

            if state.can_act(ActionRaise) and eq > raise_th and score > (0.54 if sticky else 0.48):
                amt = self._post_raise_amount(state, eq, rel_price, wet)
                if sticky:
                    lo, hi = state.raise_bounds
                    # Prefer smaller value/protection raises instead of polarization.
                    cap = lo + int(0.55 * max(1, hi - lo))
                    amt = int(self._clip(min(amt, cap), lo, hi))
                if medium_pot and eq < 0.72:
                    lo, hi = state.raise_bounds
                    cap = lo + int(0.44 * max(1, hi - lo))
                    amt = int(self._clip(min(amt, cap), lo, hi))
                return self._safe_raise(state, amt)
            if state.can_act(ActionCall) and eq >= call_th:
                return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
            return self._fallback_action(state)

        # Checked to us.
        if state.can_act(ActionRaise):
            value_bet = eq > (0.55 if state.street == "flop" else 0.58)
            exploit_bet = fold_rate > 0.44 and score > 0.54

            if sticky:
                # Against sticky ranges, bluff less and value-bet narrower.
                value_bet = eq > (0.60 if state.street == "flop" else 0.63)
                exploit_bet = False

            if value_bet or exploit_bet:
                amt = self._post_raise_amount(state, eq, 0.0, wet)
                if sticky and eq < 0.70:
                    lo, hi = state.raise_bounds
                    cap = lo + int(0.48 * max(1, hi - lo))
                    amt = int(self._clip(min(amt, cap), lo, hi))
                if medium_pot and eq < 0.74:
                    lo, hi = state.raise_bounds
                    cap = lo + int(0.40 * max(1, hi - lo))
                    amt = int(self._clip(min(amt, cap), lo, hi))
                return self._safe_raise(state, amt)

        if state.can_act(ActionCheck):
            return ActionCheck()
        return self._fallback_action(state)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
