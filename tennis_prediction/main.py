import os
import combine
import preprocess
import model


def main():
  print("Welcome to the tennis prediction program!")

  print("This programs run on a Random Forset Classifier model using sklearn to predict the winner of a tennis match.")

  with open("settings.txt", "r") as file:
    lines = file.readlines()
  acc = lines[2].strip()

  year_a = input("Starting year (or 'all'): ")
  if year_a == "all":
    year_b = "all"
  else:
    year_b = input("Ending year: ")

  if year_b < year_a:
    print("Invalid input, ending year cannot be smaller than starting year")
    quit()

  if year_a.lower() == "all":
    with open("settings.txt", "w") as file:
      file.write("1985\n")
      file.write("2024\n")
      file.write(f"{acc}")
    data_file = "data/preprocessed/preprocessed_all.csv"
  else:
    if year_a == year_b:
      with open("settings.txt", "w") as file:
        file.write(f"{year_a}\n")
        file.write(f"{year_a}\n")
        file.write(f"{acc}")
      data_file = f"data/preprocessed/preprocessed_{year_a}.csv"
    else:
      with open("settings.txt", "w") as file:
        file.write(f"{year_a}\n")
        file.write(f"{year_b}\n")
        file.write(f"{acc}")
      data_file = f"data/preprocessed/preprocesses_{year_a}_{year_b}.csv"

  if not os.path.exists(data_file):
    if year_a == year_b:
      print("Preprocessing new data")
      preprocess.process()
    else:
      print("Merging datasets")
      combine.add()
      print("Preprocessing new data")
      preprocess.process()
  
  def train():
    retrain = input("Would you like to re-train the model? (Y/N)")
    if retrain.lower() not in ["y", "n"]:
      print("Invalid input")
      quit()
    
    if retrain.lower() == "y":
      model.train()
      print("Model updated")
    else:
      print("Continuing")

  train()
  player1 = input("Enter the first player's name: ")
  player2 = input("Enter the second player's name: ")
  
  import predict
  result = predict.predict(player1, player2)
  
  print(f"With {acc} accuracy")

if __name__ == "__main__":
  main()