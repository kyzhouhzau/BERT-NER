import sys
import pandas as pd
import ner_evaluation


def get_results_dataframe(filename):
    """
    Parameters:
        - filename: CSV/TSV of NER output
    """
    header = ["word", "true_label", "predicted_label"]
    rows = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            rows.append(line.split(" "))

    df = pd.DataFrame(data=rows, columns=header)
    return df


def get_named_entities(df_results):
    """
    Parameters:
        - df_results: output of `get_results_dataframe`
    """
    labels = list(set(s[2:] for s in set(df_results["true_label"]) if s != "O"))
    labels.sort()
    
    result = {
        "true": ner_evaluation.collect_named_entities(df_results["true_label"]),
        "predicted": ner_evaluation.collect_named_entities(df_results["predicted_label"]),
        "labels": labels,
    }
    return result


def get_evaluation_results(named_entities):
    """
    Parameters:
        - named_entities: output of `get_named_entities`
    """
    ne_true = named_entities["true"]
    ne_predicted = named_entities["predicted"]
    labels = named_entities["labels"]
    
    header = [
        "tag",
        "total",
        "TP",
        "FP",
#         "TN",
        "FN",
        "precision",
        "recall",
        "f1",
#         "accuracy",
    ]
    rows = []
    for tag in labels:
        s_true = set([e for e in ne_true if e.e_type == tag])
        s_predicted = set([e for e in ne_predicted if e.e_type == tag])

        total_true = len(s_true)
        tp = len(s_true & s_predicted)
        fp = len(s_predicted - s_true)
        tn = None
        fn = total_true - tp # total_true = TP + FN

        p = 1. * tp / (tp + fp) if (tp + fp) > 0. else 0.
        r = 1. * tp / total_true if total_true > 0. else 0.
        f1 = 2. * p * r / (p + r) if (p + r) > 0. else 0.
        acc = None

        rows.append([
            tag,
            total_true,
            tp,
            fp,
#             tn,
            fn,
            p,
            r,
            f1,
#             acc,
        ])
        
    df = pd.DataFrame(data=rows, columns=header)
    
    micro_avg_p = 1. * df["TP"].sum() / (df["TP"].sum() + df["FP"].sum())
    micro_avg_r = 1. * df["TP"].sum() / (df["TP"].sum() + df["FN"].sum())
    micro_avg_f1 = 2. * micro_avg_p * micro_avg_r / (micro_avg_p + micro_avg_r) if (micro_avg_p + micro_avg_r) > 0. else 0.
    
    macro_avg_p = df["precision"].mean()
    macro_avg_r = df["recall"].mean()
    macro_avg_f1 = 2. * macro_avg_p * macro_avg_r / (macro_avg_p + macro_avg_r) if (macro_avg_p + macro_avg_r) > 0. else 0.
    
    results = {
        "df": df,
        "micro-avg": {
            "precision": micro_avg_p,
            "recall": micro_avg_r,
            "f1": micro_avg_f1,
        },
        "macro-avg": {
            "precision": macro_avg_p,
            "recall": macro_avg_r,
            "f1": macro_avg_f1,
        },
    }
    
    return results


def run(filename):
    df_results = get_results_dataframe(filename)
    named_entities = get_named_entities(df_results)
    eval_results = get_evaluation_results(named_entities)
    return eval_results


if __name__ == "__main__":
    filename = sys.argv[1]
    eval_results = run(filename)
    
    print(eval_results["df"])
    print("Micro-Avg:", eval_results["micro-avg"])
    print("Macro-Avg:", eval_results["macro-avg"])
