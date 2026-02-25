import csv
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

relationships = {}
#for k in range(len(rows)):
for k in range(500):
    for i in range(22):
        for j in range(22):
            if ((i != j) and (i in ofInterest) and (j in ofInterest)):
                key = f"rel{i}_{j}"
                reverseKey = f"rel{j}_{i}"
                if ((key not in relationships) and (reverseKey not in relationships)):
                    relationships[key] = []
                if (reverseKey not in relationships):
                    relationships[key].append((rows[k][i], rows[k][j]))

for i in relationships.keys():
    relationships[i].sort()
    #print(relationships[i])
    #print("\n-----------------------------\n")

for i in relationships.keys():
    for j in relationships[i]:
        print(j)
        input("Press Enter to continue...")
