import pandas as pd
import joblib

pm = pd.read_csv("data/player_mapping.csv")
model = joblib.load("results/small_tennis_model.pkl")

def write(filename, line_number, text):
    with open(filename, "r") as file:
        lines = file.readlines()

    if line_number < 1 or line_number > len(lines):
        print("Invalid line number")
        return

    lines[line_number - 1] = text + "\n"

    with open(filename, "w") as file:
        file.writelines(lines)

def get_player_id(player_name):
    player_data = pm[pm["player_name"] == player_name]
    if player_data.empty:
        print(f"Player with ID {player_name} not found in the dataset.")
        return None
    return player_data.iloc[0]["player_id"]

def get_stats(player_name):
    with open("settings.txt", "r") as file:
        lines = file.readlines()

    a = lines[0].strip()
    b = lines[1].strip()

    if a == b:
        df = pd.read_csv(f"data/preprocessed/preprocessed_{a}.csv")
    elif int(a) == 1985 and int(b) == 2024:
        df = pd.read_csv("data/preprocessed/preprocessed_all.csv")
    else:
        df = pd.read_csv(f"data/preprocessed/preprocessed_{a}_{b}.csv")

    player_id = get_player_id(player_name)
    if player_id is None:
        return None

    player_data = df[df["player_id"] == player_id].drop(columns=["player_id", "opponent_id", "winner"])
    if player_data.empty:
        print(f"Player {player_name} not found in the dataset.")
        return None
    return player_data.iloc[-1]

def predict(player1_name, player2_name):
    player1_stats = get_stats(player1_name)
    player2_stats = get_stats(player2_name)

    if player1_stats is None or player2_stats is None:
        print("Error: Player data not found.")
        return None

    match_data = pd.DataFrame([{ #! DO NOT TOUCH
        "player_ace": player1_stats["player_ace"], "player_df": player1_stats["player_df"],
        "player_bpSaved": player1_stats["player_bpSaved"], "player_bpFaced": player1_stats["player_bpFaced"],
        "opponent_ace": player2_stats["player_ace"], "opponent_df": player2_stats["player_df"],
        "opponent_bpSaved": player2_stats["opponent_bpSaved"], "opponent_bpFaced": player2_stats["opponent_bpFaced"],
        "player_rank": player1_stats["player_rank"], "player_rank_points": player1_stats["player_rank_points"],
        "opponent_rank": player2_stats["player_rank"], "opponent_rank_points": player2_stats["opponent_rank_points"],
        "surface_Carpet": player1_stats["surface_Carpet"], "surface_Clay": player1_stats["surface_Clay"],
        "surface_Grass": player1_stats["surface_Grass"], "surface_Hard": player1_stats["surface_Hard"],
        "tourney_level_A": player1_stats["tourney_level_A"], "tourney_level_D": player1_stats["tourney_level_D"],
        "tourney_level_F": player1_stats["tourney_level_F"], "tourney_level_G": player1_stats["tourney_level_G"],
        "tourney_level_M": player1_stats["tourney_level_M"], "best_of_3": player1_stats["best_of_3"],
        "best_of_5": player1_stats["best_of_5"]
    }])

    prediction = model.predict(match_data)[0]
    winner = player1_name if prediction == 1 else player2_name
    print(f"\n🏆 {player1_name} vs {player2_name} → Winner: {winner}")
    return winner

def run_tournament():
    with open("tournament.txt", "r") as file:
        players = [line.strip() for line in file if line.strip()]

    if len(players) % 2 != 0:
        print("Odd number of players. Please provide an even number.")
        return

    round_num = 1
    while len(players) > 1:
        print(f"\n=== Round {round_num} ===")
        next_round = []
        for i in range(0, len(players), 2):
            winner = predict(players[i], players[i + 1])
            if winner:
                next_round.append(winner)

        players = next_round
        round_num += 1

    print(f"\n🎉 Final Winner of the Tournament: {players[0]} 🎉\n")

if __name__ == "__main__":
    run_tournament()
