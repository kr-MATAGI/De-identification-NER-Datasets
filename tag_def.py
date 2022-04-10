from dataclasses import dataclass, field
from typing import List

DE_IDENT_TAG = {
    "X": 0, "O": 1,
    "B-PER": 2, "I-PER": 3, # Person
    "B-ORG": 4, "I-ORG": 5, # Organization
    "B-NUM": 6, "I-NUM": 7, # Number / Numeric
    "B-LOC": 8, "I-LOC": 9, # Location
    "B-POS": 10, "I-POS": 11, # Position
    "B-DAT": 12, "I-DAT": 13, # Date
    "B-HEC": 14, "I-HEC": 15, # Health Condition
    "B-INF": 16, "I-INF": 17, # Information
    "B-EVT": 18, "I-EVT": 19, # Event
    "B-PRI": 20, "I-PRI": 21, # Prize
    "B-AFW": 22, "I-AFW": 23, # Artifacts
    "B-PIV": 24, "I-PIV": 25 # Private
}

@dataclass
class DE_IDENT_ZIP:
    title: str = ""
    sent: str = ""
    segment_list: List[str] = field(default_factory=list)