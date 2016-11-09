import csv
import datetime

DATE_IDX = -3
RAIN_IDX = -2
SNOW_IDX = -1
FORMAT = "%Y-%m-%d"


def parsedate(x):
    if not isinstance(x, datetime.datetime):
        x = datetime.datetime.strptime(x, FORMAT)
    return x


def read_file(filename):
    """
    Read file and store lines into array for later processing.
    """
    f = open(filename, "rt")
    reader = csv.reader(f, delimiter=",")
    next(reader) # skip header line
    return list(reader)

def parse_arso_data():
    result = {}
    prev_date = ""
    data1 = read_file("..\\preprocess\\bezigrad.csv")
    data2 = read_file("..\\preprocess\\dobrunje.csv")
    for row1, row2 in zip(data1, data2):
        if len(row1) != 0 and len(row2)  != 0 and prev_date != "":
            padavine = (float(row1[RAIN_IDX]) + float(row2[RAIN_IDX])) / float(2)
            sneg = (float(row1[SNOW_IDX]) + float(row2[SNOW_IDX])) / float(2)
            result[prev_date] = [padavine, sneg]
        elif len(row1) != 0 and len(row2)  != 0:
            prev_date = row1[DATE_IDX]
    return result

