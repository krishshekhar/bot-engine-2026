"""
Thin adapter module that exposes the `Player` class from
`botharshu_v4.2.py` under an import‑friendly module name.
"""

import os
from importlib.machinery import SourceFileLoader


_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_PATH = os.path.join(_HERE, "botharshu_v4.2.py")
_MOD = SourceFileLoader("botharshu_v4_2_src", _SRC_PATH).load_module()

# Re‑export the Player class so `from botharshu_v4_2 import Player`
# works as expected.
Player = _MOD.Player

