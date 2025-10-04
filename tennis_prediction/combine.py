import pandas as pd
import os

def add():
  with open("settings.txt", "r") as file:
    lines = file.readlines()

  a = lines[0].strip()
  b = lines[1].strip()

  years = list(range(int(a), int(b)))

  dfs = []

  for year in years:
    file_path = f"data/atp/atp_matches_{year}.csv"

    if os.path.exists(file_path):

      df = pd.read_csv(file_path)

      df['year'] = year

      dfs.append(df)

    else:

      print(f"File not found: {file_path}")

  df = pd.concat(dfs, ignore_index=True)

  df.to_csv(f"data/atp/groups/atp_matches_{a}_{b}.csv", index=False)

  print(f"Data saved as 'atp_matches_{a}_{b}.csv' in 'data' folder")

if __name__ == "__main__":
  add() 