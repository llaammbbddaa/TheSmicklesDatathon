import csv
from collections import Counter
filename = "globalHealthStats.csv"  # File name
fields = []  # Column names
rows = []    # Data rows

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)  # Reader object

    fields = next(csvreader)  # Read header
    for row in csvreader:     # Read rows
        rows.append(row)

    #print("Total no. of rows: %d" % csvreader.line_num)  # Row count
    #print("Total no. of columns: %d" % len(fields))  # Column count

# country, disease name, disease category, gender, age group, treatement type
ofInterest = [0, 2, 3, 7, 8, 13]
TOP_N = 10

pair_counts = {}

for row in rows:
    for idx_i, col_i in enumerate(ofInterest):
        for col_j in ofInterest[idx_i + 1:]:
            key = (col_i, col_j)
            if key not in pair_counts:
                pair_counts[key] = Counter()
            pair_counts[key][(row[col_i], row[col_j])] += 1

print("Relationship data (top combinations per variable pair):")
for (col_i, col_j), counter in pair_counts.items():
    name_i = fields[col_i]
    name_j = fields[col_j]
    print(f"\n{name_i} <-> {name_j}")
    for (val_i, val_j), count in counter.most_common(TOP_N):
        print(f"  {val_i} | {val_j} -> {count}")

print("\nConclusions (most frequent combinations per pair):")
for (col_i, col_j), counter in pair_counts.items():
    if not counter:
        continue
    (val_i, val_j), count = counter.most_common(1)[0]
    name_i = fields[col_i]
    name_j = fields[col_j]
    print(f"- {name_i} + {name_j}: {val_i} / {val_j} occurs {count} times")
