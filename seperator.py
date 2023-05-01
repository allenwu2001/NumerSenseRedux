import sys
import json

input_filename = sys.argv[1]
masked_output_filename = sys.argv[2]
truth_output_filename = sys.argv[3]

print(
    f"Input: {input_filename}\nMasked: {masked_output_filename}\nTruth: {truth_output_filename}"
)

with open(input_filename) as f:
    data = f.read().splitlines()

masked_sents = []
truth_sents = []
for line in data:
    temp = line.split("\t")
    masked_sent = temp[0] + "\n"
    fill_word = temp[1]
    truth_sent = masked_sent.replace("<mask>", fill_word)

    masked_sents.append(masked_sent)
    truth_sents.append(truth_sent)

with open(masked_output_filename, "w") as f:
    for masked_sent in masked_sents:
        f.write(masked_sent)

with open(truth_output_filename, "w") as f:
    for truth_sent in truth_sents:
        f.write(truth_sent)
