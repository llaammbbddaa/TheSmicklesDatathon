import csv
filename = "globalHealthStats/globalHealthStats.csv"  # File name
fields = []  # Column names
rows = []    # Data rows

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)  # Reader object

    fields = next(csvreader)  # Read header
    for row in csvreader:     # Read rows
        rows.append(row)

    #print("Total no. of rows: %d" % csvreader.line_num)  # Row count
    #print("Total no. of columns: %d" % len(fields))  # Column count


ofInterest = [0, 1, 2, 3, 7, 8, 13]

relationships = {}
#for k in range(len(rows)):
for k in range(5):
    for i in range(min(22, len(rows[0]))):
        for j in range(min(22, len(rows[0]))):
            if ((i != j) and (i in ofInterest) and (j in ofInterest)):
                key = f"rel{i}_{j}"
                if key not in relationships:
                    relationships[key] = []
                relationships[key].append((rows[k][i], rows[k][j]))

for i in relationships.keys():
    relationships[i].sort()
    print(relationships[i])
    print("\n  -----------------------------\n")