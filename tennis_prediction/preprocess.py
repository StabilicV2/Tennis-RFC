#cSpell:disable
import pandas as pd
import numpy as np

USEFUL_COLUMNS = [
    "tourney_date", "surface", "tourney_level", "best_of", "winner_id",
    "loser_id", "winner_rank", "loser_rank", "winner_rank_points",
    "loser_rank_points", "w_ace", "w_df", "w_bpFaced",
    "l_ace", "l_df", "l_bpFaced", "w_bpSaved", "l_bpSaved",
    "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "winner_age", "loser_age", "winner_ht", "loser_ht",
  ]

def load(file_path):
  df = pd.read_csv(file_path, usecols=USEFUL_COLUMNS)
  df["date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")

  df["match_id"] = df.apply(
    lambda x: "_".join(sorted([str(x["winner_id"]), str(x["loser_id"])])) + "_" + str(x["tourney_date"]),
    axis=1
  )

  df = df.dropna(subset=["winner_rank", "loser_rank"])
  df = pd.get_dummies(df, columns=["surface", "tourney_level", "best_of"])
  df["pair_key"] = df.apply(
    lambda x: "_".join(sorted([str(int(x["winner_id"])), str(int(x["loser_id"]))])),
    axis=1
  )
  return df

def transform(df):

  df_winner = df.copy()
  df_loser = df.copy()

  #! Regular features

  df_winner = df_winner.rename(columns={
    "winner_id": "player_id", "loser_id": "opponent_id",
    "winner_rank": "player_rank", "loser_rank": "opponent_rank",
    "winner_rank_points": "player_rank_points", "loser_rank_points": "opponent_rank_points",
    "w_ace": "player_ace", "w_df": "player_df", "w_bpSaved": "player_bpSaved", "w_bpFaced": "player_bpFaced",
    "l_ace": "opponent_ace", "l_df": "opponent_df", "l_bpSaved": "opponent_bpSaved", "l_bpFaced": "opponent_bpFaced",
    "w_svpt": "player_svpt", "w_1stIn": "player_1stIn", "w_1stWon": "player_1stWon", "w_2ndWon": "player_2ndWon", "w_SvGms": "player_SvGms",
    "l_svpt": "opponent_svpt", "l_1stIn": "opponent_1stIn", "l_1stWon": "opponent_1stWon", "l_2ndWon": "opponent_2ndWon", "l_SvGms": "opponent_SvGms",
    "winner_age": "player_age", "loser_age": "opponent_age",
    "winner_ht": "player_ht", "loser_ht": "opponent_ht"
  })

  df_loser = df_loser.rename(columns={
    "loser_id": "player_id", "winner_id": "opponent_id",
    "loser_rank": "player_rank", "winner_rank": "opponent_rank",
    "loser_rank_points": "player_rank_points", "winner_rank_points": "opponent_rank_points",
    "l_ace": "player_ace", "l_df": "player_df", "l_bpSaved": "player_bpSaved", "l_bpFaced": "player_bpFaced",
    "w_ace": "opponent_ace", "w_df": "opponent_df", "w_bpSaved": "opponent_bpSaved", "w_bpFaced": "opponent_bpFaced",
    "l_svpt": "player_svpt", "l_1stIn": "player_1stIn", "l_1stWon": "player_1stWon", "l_2ndWon": "player_2ndWon", "l_SvGms": "player_SvGms",
    "w_svpt": "opponent_svpt", "w_1stIn": "opponent_1stIn", "w_1stWon": "opponent_1stWon", "w_2ndWon": "opponent_2ndWon", "w_SvGms": "opponent_SvGms",
    "loser_age": "player_age", "winner_age": "opponent_age",
    "loser_ht": "player_ht", "winner_ht": "opponent_ht"
  })
    
  df_winner["winner"] = 1
  df_loser["winner"] = 0

  shared_cols = ["match_id", "date"]
  df_winner[shared_cols] = df[shared_cols]
  df_loser[shared_cols] = df[shared_cols]

  df_final = pd.concat([df_winner, df_loser], ignore_index=True)

  #! Engineered features

  #? Serve percentages
  df_final["player_first_serve_pct"] = df_final["player_1stIn"] / df_final["player_svpt"]
  df_final["opponent_first_serve_pct"] = df_final["opponent_1stIn"] / df_final["opponent_svpt"]

  df_final["player_first_serve_win_pct"] = df_final["player_1stWon"] / df_final["player_1stIn"]
  df_final["opponent_first_serve_win_pct"] = df_final["opponent_1stWon"] / df_final["opponent_1stIn"]

  df_final["player_second_serve_win_pct"] = df_final["player_2ndWon"] / (df_final["player_svpt"] - df_final["player_1stIn"])
  df_final["opponent_second_serve_win_pct"] = df_final["opponent_2ndWon"] / (df_final["opponent_svpt"] - df_final["opponent_1stIn"])

  df_final["player_bp_save_pct"] = df_final["player_bpSaved"] / df_final["player_bpFaced"]
  df_final["opponent_bp_save_pct"] = df_final["opponent_bpSaved"] / df_final["opponent_bpFaced"]

  #? Difference features
  df_final["rank_diff"] = df_final["opponent_rank"] - df_final["player_rank"]
  df_final["rank_points_diff"] = df_final["opponent_rank_points"] - df_final["player_rank_points"]
  df_final["ace_diff"] = df_final["player_ace"] - df_final["opponent_ace"]
  df_final["df_diff"] = df_final["player_df"] - df_final["opponent_df"]
  df_final["first_serve_win_diff"] = df_final["player_first_serve_win_pct"] - df_final["opponent_first_serve_win_pct"]
  df_final["bp_save_diff"] = df_final["player_bp_save_pct"] - df_final["opponent_bp_save_pct"]
  df_final["age_diff"] = df_final["player_age"] - df_final["opponent_age"]
  df_final["height_diff"] = df_final["player_ht"] - df_final["opponent_ht"]

  #? Serve dominance and pressure
  df_final["serve_dom_diff"] = ((df_final["player_1stWon"] + df_final["player_2ndWon"]) / df_final["player_svpt"]) - (
      (df_final["opponent_1stWon"] + df_final["opponent_2ndWon"]) / df_final["opponent_svpt"]
  )
  df_final["bp_pressure_diff"] = (df_final["player_bpFaced"] / df_final["player_SvGms"]) - (
      df_final["opponent_bpFaced"] / df_final["opponent_SvGms"]
  )

  #? Symmetric pair key (H2H)
  df_final["pair_key"] = df_final.apply(
    lambda x: "_".join(sorted([str(x["player_id"]), str(x["opponent_id"])])), axis=1
  )

  df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)

  df_final = add_h2h(df_final)

  return df_final

def add_h2h(df):
  df = df.sort_values("date")
  h2h_matches, h2h_winrates = [], []
  history = {}

  for _, row in df.iterrows():
    key = row["pair_key"]
    player = row["player_id"]
    date = row["date"]

    if key not in history:
      history[key] = []

    #> Previous matchups only
    past_matches = [m for m in history[key] if m["date"] < date]
    total = len(past_matches)
    wins = sum(1 for m in past_matches if m["winner_id"] == player)

    h2h_matches.append(total)
    h2h_winrates.append(wins / total if total > 0 else 0)

    #> Add to history
    history[key].append({"date": date, "winner_id": player if row["winner"] == 1 else row["opponent_id"]})

  df["h2h_prev_matches"] = h2h_matches
  df["h2h_winrate_vs_opp"] = h2h_winrates
  return df

def process():
  with open("settings.txt", "r") as file:
    lines = file.readlines()

  a = lines[0].strip()
  b = lines[1].strip()
  print(a)
  print(b)
  if a == b:
    FILE_PATH = f"data/atp/atp_matches_{a}.csv"
  elif a == str(1985) and b == str(2024):
    FILE_PATH = f"data/atp/groups/atp_matches_all.csv"
  else:
    FILE_PATH = f"data/atp/groups/atp_matches_{a}_{b}.csv"
    
  df_raw = load(FILE_PATH)
  df_final = transform(df_raw)

  if a == b:
    df_final.to_csv(f"data/preprocessed/preprocessed_{a}.csv", index=False)
    print(f"Data Preprocessing Complete, saved as 'preprocessed_{a}.csv' in 'data/preprocessed' folder")
    print(df_final.head())
  elif a == str(1985) and b == str(2024):
    df_final.to_csv(f"data/preprocessed/preprocessed_all.csv", index=False)
    print(f"Data Preprocessing Complete, saved as 'preprocessed_all.csv' in 'data/preprocessed' folder")
    print(df_final.head())
  else:
    df_final.to_csv(f"data/preprocessed/preprocessed_{a}_{b}.csv", index=False)
    print(f"Data Preprocessing Complete, saved as 'preprocessed_{a}_{b}.csv' in 'data/preprocessed' folder")
    print(df_final.head())

if __name__ == "__main__":
  process()