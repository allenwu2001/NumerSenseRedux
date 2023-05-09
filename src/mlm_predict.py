from happytransformer import HappyXLNET
from happytransformer import HappyROBERTA
from happytransformer import HappyBERT
from happytransformer import HappyDISTILBERT
from happytransformer import HappyALBERT
from happytransformer import HappyDEBERTA
from tqdm import tqdm
import sys
import json

num_list = [
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "no",
    "zero",
    "one",
]

if __name__ == "__main__":
    model_str = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]
    model = None
    if model_str.startswith("reload_"):
        if model_str.startswith("reload_bert"):
            model_str = model_str.replace("reload_bert:", "")
            model = HappyBERT(model_str)
        elif model_str.startswith("reload_roberta"):
            model_str = model_str.replace("reload_roberta:", "")
            model = HappyROBERTA(model_str)
        elif model_str.startswith("reload_distilbert"):
            model_str = model_str.replace("reload_distilbert:", "")
            model = HappyDISTILBERT(model_str)
        elif model_str.startswith("reload_albert"):
            model_str = model_str.replace("reload_albert:", "")
            model = HappyALBERT(model_str)
        elif model_str.startswith("reload_deberta"):
            model_str = model_str.replace("reload_deberta:", "")
            model = HappyDEBERTA(model_str)
    else:
        if model_str.startswith("bert"):
            # bert-base, bert-large
            model = HappyBERT(model_str + "-uncased")
        elif model_str.startswith("roberta"):
            # roberta-base, roberta-large
            model = HappyROBERTA(model_str)
        elif model_str.startswith("distil"):
            # distilbert-base
            model = HappyDISTILBERT(model_str + "-uncased")
        elif model_str.startswith("albert"):
            # albert-base-v2
            model = HappyALBERT(model_str + "-v2")
        elif model_str.startswith("deberta"):
            # deberta-base
            model = HappyDEBERTA("microsoft/" + model_str)
        elif model_str.startswith("xlnet"):
            # ignore
            model = HappyXLNET(model_str + "-cased")
    assert model is not None

    with open(input_filename) as f:
        data = f.read().splitlines()
    predictions = []
    for masked_sent in tqdm(data, desc="Probing"):
        masked_sent = masked_sent.strip()
        result_list = model.predict_mask(
            masked_sent, options=num_list, num_results=1000
        )
        result_list.sort(key=lambda x: x["softmax"], reverse=True)
        # cast to float
        result_list = [
            {"word": x["word"].lower(), "score": float(x["softmax"])}
            for x in result_list
        ]
        output_str = json.dumps(
            dict(probe=masked_sent, result_list=result_list)
        )
        predictions.append(output_str)
    with open(output_filename, "w") as f:
        print("Saving resutls to", f.name)
        f.write("\n".join(predictions) + "\n")
