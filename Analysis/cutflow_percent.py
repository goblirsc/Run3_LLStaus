
import json

input_file = "cutflows.json"
output_file = "cutflow_percentages_DITAU_trigger.txt"

with open(input_file) as f:
    data = json.load(f)

with open(output_file, "w") as f:

    for process, categories in data.items():

        cuts = categories["all"]
        initial = cuts["BeforeCuts"]

        f.write(f"\n{process}\n")
        f.write(
            f"{'Cut':<30} {'Events':>10} {'Total %':>10}\n"
        )
        f.write("-" * 52 + "\n")

        for cut, count in cuts.items():

            total_eff = 100 * count / initial if initial else 0

            f.write(
                f"{cut:<30} {count:>10.0f} {total_eff:>10.2f}\n"
            )

        f.write("\n")

print(f"Tables saved to {output_file}")
