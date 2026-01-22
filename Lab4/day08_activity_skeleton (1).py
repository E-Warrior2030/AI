"""
Day 8 Activity: Duplicates Practice
Tasks:
1) Remove exact full-row duplicates
2) Apply uniqueness rule: (user, day, product)
3) Aggregate to user-level features
"""

import pandas as pd

# Sample transaction events
rows = [
    {"user": "U1", "day": "2024-01-01", "product": "A", "clicked": 1},
    {"user": "U1", "day": "2024-01-01", "product": "A", "clicked": 1},
    {"user": "U1", "day": "2024-01-01", "product": "B", "clicked": 0},
    {"user": "U2", "day": "2024-01-02", "product": "A", "clicked": 1},
]

df = pd.DataFrame(rows)

# TODO: Remove exact duplicates
# TODO: Define uniqueness rule and deduplicate by subset
# TODO: Aggregate per user: event_count, ever_clicked
