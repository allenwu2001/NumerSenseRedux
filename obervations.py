# Import the necessary modules
import json
import sys
import pandas as pd


truth_file_name = sys.argv[1]
results_file_name = sys.argv[2]
# Open the file
with open(truth_file_name, "r") as truth_file:
    # Read the lines from the file
    truth_lines = truth_file.readlines()

    # Convert the lines to a JSON object
    df = pd.read_json(path_or_buf=results_file_name, lines=True)

    hist = {}
    for i, truth_line in enumerate(truth_lines):
        json_obj = df.iloc[i]
        json_obj["result_list"].sort(
            key=lambda x: x["score"], reverse=True
        )
        print(truth_lines[i].strip())
        print(
            json_obj["probe"]
            .replace("<mask>", json_obj["result_list"][0]["word"])
            .strip()
        )
        print(sum(map(lambda x: x["score"], json_obj["result_list"])))
        print()

        if json_obj["result_list"][0]["word"] in hist:
            hist[json_obj["result_list"][0]["word"]] += 1
        else:
            hist[json_obj["result_list"][0]["word"]] = 1
print(hist)
