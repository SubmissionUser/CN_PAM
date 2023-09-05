# Helper script to calculate the difference from the selected baseline.
# This will print the DeltaMetric , DeltaTime in percentage terms

import pandas as pd

res = pd.read_csv("./gc_results.csv")

datasets = res["Dataset"].unique()

grouped = (
    res.groupby(["Dataset", "Model"])[["F1_macro", "F1_micro", "Acc", "Time"]]
    .mean()
    .sort_values("F1_macro", ascending=False)
)
grouped.reset_index()

sota_name = "WL-OA"
competitors = ["PP-1", "PP-1+VH"]
metric_score = "F1_macro"
points_to_show = []
for dataset in datasets:
    subset = grouped.reset_index()[
        grouped.reset_index()["Dataset"] == dataset
    ].set_index("Model")
    time_sota = subset.loc[sota_name]["Time"]
    perf_sota = subset.loc[sota_name][metric_score]

    for comp_name in competitors:
        time_cur = subset.loc[comp_name]["Time"]
        perf_cur = subset.loc[comp_name][metric_score]

        time_perc_dif = 100 * (time_cur - time_sota) / time_sota
        perf_perc_dif = 100 * (perf_cur - perf_sota) / perf_sota
        points_to_show.append(
            {
                "Dataset": dataset,
                "Model": comp_name,
                "Time WL-OA": time_sota,
                "Perf WL-OA": perf_sota,
                f"Time": time_cur,
                f"Perf": perf_cur,
                f"Time % Change": time_perc_dif,
                f"Perf % Change": perf_perc_dif,
            }
        )
df_plot = pd.DataFrame(points_to_show)
print(df_plot[["Dataset", "Model", "Perf % Change", "Time % Change"]].to_string())
print()
