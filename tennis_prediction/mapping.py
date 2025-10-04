#cSpell:disable
import pandas as pd

with open("settings.txt", "r") as file:
  lines = file.readlines()

a = lines[0].strip()
b = lines[1].strip()

def map():
  if a == b:
    df_raw = pd.read_csv(f"data/atp/atp_matches_{a}.csv")
  else:
    df_raw = pd.read_csv(f"data/atp/groups/atp_matches_{a}_{b}.csv")

  df_raw_cleaned = df_raw.dropna(subset=["winner_id", "winner_name", "loser_id", "loser_name"])

  winner_mapping = df_raw_cleaned[["winner_id", "winner_name"]].rename(columns={"winner_id": "player_id", "winner_name": "player_name"})
  loser_mapping = df_raw_cleaned[["loser_id", "loser_name"]].rename(columns={"loser_id": "player_id", "loser_name": "player_name"})

  player_mapping = pd.concat([winner_mapping, loser_mapping])

  player_mapping = player_mapping.drop_duplicates()

  player_mapping.to_csv("data/player_mapping.csv", index=False)

  print("Player mapping saved as 'player_mapping.csv' in 'data' folder")

if __name__ == "__main__":
  pass