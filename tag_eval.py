import sklearn.metrics as sk_m
import pandas as pd

header = ["word", "true_label", "predicted_label"]
rows = []
with open("our_en_predicted_results.txt") as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue

        rows.append(line.split(" "))

df = pd.DataFrame(data=rows, columns=header)
# not_O = df["true_label"] != "O"
# df = df[not_O]

labels = df["true_label"].drop_duplicates().tolist()
labels.remove("O")

m = sk_m.confusion_matrix(df["true_label"], df["predicted_label"], labels=labels)

df_cm = pd.DataFrame(data=m, columns=labels, index=labels)

print(df_cm)
print(sk_m.classification_report(df["true_label"], df["predicted_label"], labels=labels, digits=6))

print("MACRO_AVG f1-score: {:01.6f}".format(sk_m.f1_score(df["true_label"], df["predicted_label"], labels=labels, average='macro')))

print("MICRO_AVG f1-score: {:01.6f}".format(sk_m.f1_score(df["true_label"], df["predicted_label"], labels=labels, average='micro')))
