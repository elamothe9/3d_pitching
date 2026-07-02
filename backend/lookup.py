from pybaseball import playerid_lookup

def get_mlbam_id(first: str, last: str):
    """
    Look up a player's MLBAM ID using pybaseball.
    Example: get_mlbam_id("Gerrit", "Cole") -> 543037
    """
    df = playerid_lookup(last, first)  # note: order is (last, first)
    if df.empty:
        return None
    
    # prefer players who actually appeared in MLB
    df = df[df["mlb_played_last"].notna()]
    
    # take the most recent MLB player
    row = df.sort_values("mlb_played_last", ascending=False).iloc[0]
    return int(row.key_mlbam)

if __name__ == "__main__":
    # quick test
    first, last = "Paul", "Skenes"
    pid = get_mlbam_id(first, last)
    print(f"{first} {last} → {pid}")
