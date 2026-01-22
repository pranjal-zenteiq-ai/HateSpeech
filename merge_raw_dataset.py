import json
import pandas as pd


def merge_raw_dataset(
    map_csv="map.csv",
    output_jsonl="output.jsonl",
    error_jsonl="error.jsonl",
    out_csv="merged_raw_moderation_dataset.csv"
):
    # ----------------------------
    # Load base text map
    # ----------------------------
    df_map = pd.read_csv(map_csv)
    assert {"custom_id", "text"}.issubset(df_map.columns)

    # ----------------------------
    # Parse output.jsonl
    # ----------------------------
    rows = []

    with open(output_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id")

            row = {
                "custom_id": cid,
                "has_error": False
            }

            resp = obj.get("response", {})
            body = resp.get("body", {})
            results = body.get("results", [])

            if results:
                r0 = results[0]
                row["flagged"] = r0.get("flagged")

                # categories
                categories = r0.get("categories", {})
                for k, v in categories.items():
                    row[f"category_{k}"] = v

                # category scores
                scores = r0.get("category_scores", {})
                for k, v in scores.items():
                    row[f"category_score_{k}"] = v
            else:
                row["flagged"] = None

            rows.append(row)

    df_out = pd.DataFrame(rows)

    # ----------------------------
    # Parse error.jsonl (mark errors)
    # ----------------------------
    error_ids = set()
    with open(error_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            if cid is not None:
                error_ids.add(cid)

    df_out.loc[df_out["custom_id"].isin(error_ids), "has_error"] = True

    # ----------------------------
    # Merge everything (NO DROPS)
    # ----------------------------
    df_final = df_map.merge(df_out, on="custom_id", how="left")

    # ----------------------------
    # Save
    # ----------------------------
    df_final.to_csv(out_csv, index=False)

    print("Merged raw dataset saved to:", out_csv)
    print("Shape:", df_final.shape)
    print("\nFlagged value counts (including NaN):")
    print(df_final["flagged"].value_counts(dropna=False))
    print("\nError rows:", df_final["has_error"].sum())


if __name__ == "__main__":
    merge_raw_dataset()
