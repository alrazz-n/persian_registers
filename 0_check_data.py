import pandas as pd

# Load the CSV file
df = pd.read_csv("data/persian.csv")

# Check for duplicate IDs
duplicate_ids = df[df.duplicated(subset=["id"], keep=False)]

if len(duplicate_ids) > 0:
    print(f"YES - Found {duplicate_ids['id'].nunique()} duplicate IDs")
    print(f"Total rows with duplicates: {len(duplicate_ids)}")

    # Check if duplicates have the same values for Turku_NLP and Turku_NLP_sub
    print("\n" + "=" * 60)
    print("Checking consistency of Turku_NLP and Turku_NLP_sub columns:")
    print("=" * 60)

    inconsistent_duplicates = []

    for dup_id in duplicate_ids["id"].unique():
        dup_rows = df[df["id"] == dup_id]

        # Check if all values in Turku_NLP are the same
        turku_nlp_same = dup_rows["Turku_NLP"].nunique() == 1
        turku_nlp_sub_same = dup_rows["Turku_NLP_sub"].nunique() == 1

        if not (turku_nlp_same and turku_nlp_sub_same):
            inconsistent_duplicates.append(dup_id)
            print(f"\nID {dup_id} has INCONSISTENT values:")
            print(dup_rows[["id", "Turku_NLP", "Turku_NLP_sub"]])

    print("\n" + "=" * 60)
    if len(inconsistent_duplicates) == 0:
        print("✓ ALL duplicate IDs have consistent Turku_NLP and Turku_NLP_sub values")
    else:
        print(
            f"✗ {len(inconsistent_duplicates)} duplicate IDs have INCONSISTENT values"
        )
        print(f"Inconsistent IDs: {inconsistent_duplicates}")

else:
    print("NO - No duplicate IDs found")
    print(f"Total unique IDs: {df['id'].nunique()}")
    print(f"Total rows: {len(df)}")
