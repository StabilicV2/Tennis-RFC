from difflib import get_close_matches
import pandas as pd

def find_player_id(name, mapping_df, threshold=0.7):

  name = name.strip().lower()
  mapping_df["name_lower"] = mapping_df["player_name"].str.lower()

  exact = mapping_df.loc[mapping_df["name_lower"] == name]
  if not exact.empty:
    row = exact.iloc[0]
    return row["player_id"], row["player_name"]
  
  partial = mapping_df[mapping_df["name_lower"].str.contains(name)]
  if not partial.empty:
    row = partial.iloc[0]
    print(f"Interpeted {name} --> {row['player_name']} (Partial match)")
    return row["player_id"], row["player_name"]
  
  all_names = mapping_df["player_name"].tolist()
  close = get_close_matches(name.title(), all_names, n=3)
  if close:
    matched = close[0]
    row = mapping_df[mapping_df["player_name"] == matched].iloc[0]
    print(f"Autocorrected {name} --> {matched} (Fuzzy match)")
    return row["player_id"], matched
  
  raise ValueError(f"No player found, {name}")