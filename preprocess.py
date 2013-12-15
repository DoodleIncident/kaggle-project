import csv

with open('train.csv', 'rb') as tf,\
        open('pre/input.csv', 'wb') as pi,\
        open('pre/output.csv', 'wb') as po:
    r = csv.reader(tf, delimiter=',')
    wi = csv.writer(pi, delimiter=',')
    wo = csv.writer(po, delimiter=',')

    all_fields = list(r)

    pre_input = map(lambda row: [row[1]], all_fields)
    pre_output = map(lambda row: row[4:], all_fields)

    for row_in, row_out in zip(pre_input, pre_output):
        wi.writerow(row_in)
        wo.writerow(row_out)