#cSpell:disable
import pandas as pd
import joblib
import numpy as np

pm = pd.read_csv("data/player_mapping.csv")
model, threshold = joblib.load("results/final_tennis_model.pkl")

def get_player_id(player_name):
  player_data = pm[pm["player_name"] == player_name]

  if player_data.empty:
    print(f"Player with ID {player_name} not found in the dataset.")
    return None
  return player_data.iloc[0]["player_id"]
  
def get_dataset():
    with open("settings.txt", "r") as file:
      lines = file.readlines()

    a = lines[0].strip()
    b = lines[1].strip()

    if a == b:
      df = pd.read_csv(f"data/preprocessed/preprocessed_{a}.csv")
    if int(a) == 1985 and int(b) == 2024:
      df = pd.read_csv("data/preprocessed/preprocessed_all.csv")
    else:
      df = pd.read_csv(f"data/preprocessed/preprocessed_{a}_{b}.csv")

    return df

def get_latest_stats(df, player_name):
  player_id = get_player_id(player_name)
  if player_id is None:
    return None
  
  player_data = df[df["player_id"] == player_id]
  if player_data.empty:
    return None

  return player_data.iloc[-1] #- Most recent stats

def compute_h2h(df, player1_id, player2_id):
  #- Compute h2h features for two players
  pair_key = "_".join(sorted([str(player1_id), str(player2_id)]))
  past_matches = df[df["pair_key"] == pair_key]

  total = len(past_matches)
  if total == 0:
    return 0, 0.0
  
  #> Number of times player 1 won
  player1_wins = len(past_matches[
    (past_matches["player_id"] == player1_id) & (past_matches["winner"] == 1)
  ])

  winrate = player1_wins / total
  return total, winrate

def predict(player1_name, player2_name):
  df = get_dataset()
  player1_stats = get_latest_stats(df, player1_name)
  player2_stats = get_latest_stats(df, player2_name)

  if player1_stats is None or player2_stats is None:
    print("Error: Player data not found.")
    return None
  
  player1_id = player1_stats["player_id"]
  player2_id = player2_stats["player_id"]


  #? Compute H2H stats
  h2h_matches, h2h_winrate = compute_h2h(df, player1_id, player2_id)
  
  match_data = pd.DataFrame([{ #! DO NOT TOUCH
    "player_ace": player1_stats["player_ace"],
    "player_df": player1_stats["player_df"],
    "player_bpSaved": player1_stats["player_bpSaved"],
    "player_bpFaced": player1_stats["player_bpFaced"],
    "opponent_ace": player2_stats["player_ace"],
    "opponent_df": player2_stats["player_df"],
    "opponent_bpSaved": player2_stats["player_bpSaved"],
    "opponent_bpFaced": player2_stats["opponent_bpFaced"],
    "player_rank": player1_stats["player_rank"],
    "player_rank_points": player1_stats["player_rank_points"],
    "opponent_rank": player2_stats["player_rank"],
    "opponent_rank_points": player2_stats["player_rank_points"],
    
    "surface_Carpet": player1_stats["surface_Carpet"],
    "surface_Clay": player1_stats["surface_Clay"],
    "surface_Grass": player1_stats["surface_Grass"],
    "surface_Hard": player1_stats["surface_Hard"],
    "tourney_level_A": player1_stats["tourney_level_A"],
    "tourney_level_D": player1_stats["tourney_level_D"],
    "tourney_level_F": player1_stats["tourney_level_F"],
    "tourney_level_G": player1_stats["tourney_level_G"],
    "tourney_level_M": player1_stats["tourney_level_M"],
    "best_of_3": player1_stats["best_of_3"],
    "best_of_5": player1_stats["best_of_5"],

    "player_svpt": player1_stats["player_svpt"],
    "player_1stIn": player1_stats["player_1stIn"],
    "player_1stWon": player1_stats["player_1stWon"],
    "player_2ndWon": player1_stats["player_2ndWon"],
    "opponent_svpt": player2_stats["player_svpt"],
    "opponent_1stIn": player2_stats["player_1stIn"],
    "opponent_1stWon": player2_stats["player_1stWon"],
    "opponent_2ndWon": player2_stats["player_2ndWon"],
    "player_age": player1_stats["player_age"],
    "opponent_age": player2_stats["player_age"],
    "player_ht": player1_stats["player_ht"],
    "opponent_ht": player2_stats["player_ht"],
    "player_SvGms": player1_stats["player_SvGms"],
    "opponent_SvGms": player2_stats["player_SvGms"],
  }])

  match_data["player_first_serve_pct"] = match_data["player_1stIn"] / match_data["player_svpt"]
  match_data["opponent_first_serve_pct"] = match_data["opponent_1stIn"] / match_data["opponent_svpt"]
  match_data["player_first_serve_win_pct"] = match_data["player_1stWon"] / match_data["player_1stIn"]
  match_data["opponent_first_serve_win_pct"] = match_data["opponent_1stWon"] / match_data["opponent_1stIn"]
  match_data["player_second_serve_win_pct"] = match_data["player_2ndWon"] / (match_data["player_svpt"] - match_data["player_1stIn"])
  match_data["opponent_second_serve_win_pct"] = match_data["opponent_2ndWon"] / (match_data["opponent_svpt"] - match_data["opponent_1stIn"])
  match_data["player_bp_save_pct"] = match_data["player_bpSaved"] / match_data["player_bpFaced"]
  match_data["opponent_bp_save_pct"] = match_data["opponent_bpSaved"] / match_data["opponent_bpFaced"]
  match_data["rank_diff"] = match_data["opponent_rank"] - match_data["player_rank"]
  match_data["rank_points_diff"] = match_data["opponent_rank_points"] - match_data["player_rank_points"]
  match_data["ace_diff"] = match_data["player_ace"] - match_data["opponent_ace"]
  match_data["df_diff"] = match_data["player_df"] - match_data["opponent_df"]
  match_data["first_serve_win_diff"] = match_data["player_first_serve_win_pct"] - match_data["opponent_first_serve_win_pct"]
  match_data["bp_save_diff"] = match_data["player_bp_save_pct"] - match_data["opponent_bp_save_pct"]
  match_data["age_diff"] = match_data["player_age"] - match_data["opponent_age"]
  match_data["height_diff"] = match_data["player_ht"] - match_data["opponent_ht"]
  match_data["serve_dom_diff"] = ((match_data["player_1stWon"] + match_data["player_2ndWon"]) / match_data["player_svpt"]) - (
    (match_data["opponent_1stWon"] + match_data["opponent_2ndWon"]) / match_data["opponent_svpt"])
  match_data["bp_pressure_diff"] = (
    (match_data["player_bpFaced"] / match_data["player_SvGms"]) -
    (match_data["opponent_bpFaced"] / match_data["opponent_SvGms"]))
  
  match_data["h2h_prev_matches"] = h2h_matches
  match_data["h2h_winrate_vs_opp"] = h2h_winrate

  expected_order = list(model.feature_names_in_)
  
  match_data = match_data[expected_order]

  if not match_data.columns.equals(pd.Index(expected_order)):
    raise ValueError("Columns are still not aligned with the model!")
  
  match_data = match_data.replace([np.inf, -np.inf], np.nan).fillna(0)

  prob = model.predict_proba(match_data)[0][1]

  prediction = 1 if prob >= threshold else 0
  winner = player1_name if prediction == 1 else player2_name
  print(f"Predicted Winner: {winner}")

if __name__ == "__main__":
  pass