import csv

input_file = "../data/user_data.csv"
output_file = "data/user_ids.txt"

user_ids = []
with open(input_file, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row["user_id"].strip()
        if uid:
            user_ids.append(uid)

with open(output_file, "w") as f:
    for uid in user_ids:
        f.write(uid + "\n")

print(f"Extracted {len(user_ids)} user_ids to {output_file}")