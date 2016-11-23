import csv
import datetime
from collections import defaultdict

DATE_S_IDX = -3
DATE_E_IDX = -2
LINE_IDX = -1
FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def parsedate(x):
    if not isinstance(x, datetime.datetime):
        x = datetime.datetime.strptime(x, FORMAT)
    return x


def read_file(filename):
    """
    Read file and store lines into array for later processing.
    """
    f = open(filename, "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter=",")
    next(reader) # skip header line
    return list(reader)

def parse_detour_data():
    result = defaultdict(list)
    data = read_file("..\\preprocess\\obvozi.txt")
    for row in data:
        result[row[LINE_IDX]].append([row[DATE_S_IDX],row[DATE_E_IDX]])
    data2 = read_file("..\\preprocess\\detours2.csv")
    for row in data2:
        result[row[LINE_IDX]].append([row[DATE_S_IDX],row[DATE_E_IDX]])
    return result

def check_detour(line, date, detours):
    l_det = detours[line]
    for d in l_det:
        if parsedate(d[0]) <= parsedate(date) <= parsedate(d[1]):
            return 1
    return 0
