from datetime import date
from dataclasses import dataclass, field

@dataclass
class PositionConfig:
    ticker:         str
    name:           str
    entry_price:    float
    target_price:   float
    implied_vol:    float
    holding_years:  float
    position_date:  date
    weight:         float
    risk_free_rate: float = 0.0432

POSITIONS = [
    PositionConfig("MU",    "Micron",    85.0,  130.0, 0.45, 1.0, date(2024,6,1), 0.20),
    PositionConfig("MSFT",  "Microsoft", 415.0, 500.0, 0.22, 1.0, date(2024,6,1), 0.20),
    PositionConfig("LLY",   "Eli Lilly", 750.0, 950.0, 0.25, 1.0, date(2024,6,1), 0.20),
    PositionConfig("ASML",  "ASML",      750.0, 950.0, 0.28, 1.0, date(2024,6,1), 0.20),
    PositionConfig("MC.PA", "LVMH",      680.0, 850.0, 0.22, 1.0, date(2024,6,1), 0.20),
]