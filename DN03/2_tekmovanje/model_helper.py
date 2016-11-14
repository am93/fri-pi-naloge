from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse
import gzip, csv
import scipy.sparse as sp
import numpy as np
import lpputils
from random import shuffle
from sklearn.metrics import mean_absolute_error
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from arso_parser import parse_arso_data
from obvozi_parser import parse_detour_data
from obvozi_parser import check_detour

DEP_IDX = -3
ARR_IDX = -1
DRV_IDX =  1
BUS_IDX = 0

HOLIDAYS = ['2012-01-01', '2012-01-02', '2012-02-08', '2012-04-09', '2012-04-27', '2012-05-01', '2012-05-02',
                '2012-05-31', '2012-06-25', '2012-08-15', '2012-10-31', '2012-11-01', '2012-12-25', '2012-12-26']
SCHOOL_HOL = ['2012-02-20', '2012-02-21', '2012-02-22', '2012-02-23', '2012-02-24', '2012-04-30',
              '2012-12-24', '2012-12-25', '2012-12-26', '2012-12-27', '2012-12-28', '2012-12-31']
SUMMER_HOL = ['2012-06-25 00:00:00.000', '2012-08-31 00:00:00.000']

drivers = {}
buses = {}
driver_idxs = {}
buses_idxs = {}
arso = {}
detours = {}

def visualize(train_data, _month, day_s, day_e):
    """
    Function which outputs daily travel time by hour (graph + text)
    """
    comp_data = []
    times = np.zeros(24)
    cnts = np.zeros(24)
    for d in range(day_s,day_e,1):
        for row in train_data:
            date = lpputils.parsedate(row[DEP_IDX])
            hour = date.hour
            month = date.month
            day = date.day
            if month == _month and day == d:
                times[hour] += lpputils.tsdiff(row[ARR_IDX], row[DEP_IDX])
                cnts[hour] += 1
    norm_times = [float(times[i]) / (float(cnts[i])+0.0000000000001) for i in range(len(times))]
    #comp_data.append(np.asarray(norm_times))
    print(norm_times)
    #with open('vizualizacija.csv', 'wb') as abc:
    #    np.savetxt(abc, np.asarray(comp_data), delimiter=",", fmt="%d")

    #data = np.genfromtxt('vizualizacija.csv', delimiter=',')
    #for i in range(len(data)):
    plt.plot(norm_times, label='the data')
    plt.show()



def idx_init(data):
    """
    Function initializes lookup table for buses and drivers
    """
    counterB = 0
    counterD = 0
    for row in data:
        if row[BUS_IDX] not in buses_idxs.keys():
            buses_idxs[row[BUS_IDX]] = counterB;
            counterB += 1
        if row[DRV_IDX] not in driver_idxs.keys():
            driver_idxs[row[DRV_IDX]] = counterD
            counterD += 1


def driver_average(driver_id=None, data=None):
    global drivers
    if driver_id is None:
        for row in data:
            if row[DRV_IDX] not in drivers.keys():
                drivers[row[DRV_IDX]] = [lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX]), 1]
            else:
                tmp = drivers[row[DRV_IDX]]
                tmp[0] = lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX])
                tmp[1] += 1
                drivers[row[DRV_IDX]] = tmp
        tmp = {driver : float(drivers[driver][0]) / float(drivers[driver][1])  for driver in drivers.keys()}
        drivers = {driver: tmp[driver] / max(tmp.values()) for driver in tmp}
    else:
        try:
            return drivers[driver_id]
        except KeyError:
            print(sum(drivers.values()) / float(len(drivers)))
            return sum(drivers.values()) / float(len(drivers))


def bus_average(bus_id=None, data=None):
    global buses
    if bus_id is None:
        for row in data:
            if row[BUS_IDX] not in buses.keys():
                buses[row[BUS_IDX]] = [lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX]), 1]
            else:
                tmp = buses[row[BUS_IDX]]
                tmp[0] = lpputils.tsdiff(row[ARR_IDX],row[DEP_IDX])
                tmp[1] += 1
                buses[row[BUS_IDX]] = tmp
        tmp = {bus : float(buses[bus][0]) / float(buses[bus][1])  for bus in buses.keys()}
        buses = {bus: tmp[bus] / max(tmp.values()) for bus in tmp}
    else:
        try:
            return buses[bus_id]
        except KeyError:
            print(sum(buses.values()) / float(len(buses)))
            return sum(buses.values()) / float(len(buses))


def model1(row):
    """
    MODEL1 : norm day, norm hour, holiday, school holiday, avg. driver, avg. bus
    """
    result = np.zeros(6)
    result[0] = lpputils.parsedate(row[DEP_IDX]).weekday() / 7.0 # day
    result[1] =  lpputils.parsedate(row[DEP_IDX]).hour / 24.0 # hour
    date = lpputils.parsedate(row[DEP_IDX]).date()

    holiday = 0
    school_hol = 0
    #summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[2] = holiday
    result[3] = school_hol
    #result[4] = summer_hol
    result[4] = driver_average(row[DRV_IDX])
    result[5] = bus_average(row[BUS_IDX])

    return result


def model2(row):
    """
    MODEL2 : binary day and week attributes + holiday (binary)
    indeksi : 0-6 dnevi, 7-31 ura
    server: 152.32996
    lokalno: 129.303771
    """
    result = np.zeros(7+24+2)

    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    result[day] = 1
    result[7+hour] = 1

    date = lpputils.parsedate(row[DEP_IDX]).date()

    holiday = 0
    school_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1

    result[-2] = holiday
    result[-1] = school_hol

    return result


def model3(row):
    """
    MODEL3 : binary day and week attributes + drivers + buses + holiday
    indeksi : dnevi, ure, vozniki, busi, pocitnice
    server: ???
    lokalno: 129.2525
    """
    day_offset = 0
    hour_offset = 7
    driver_offset = 31
    buses_offset = driver_offset + len(driver_idxs)
    holiday_offset = buses_offset + len(buses_idxs)
    result = np.zeros(holiday_offset+3)

    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    result[day_offset + day] = 1
    result[hour_offset + hour] = 1
    if row[DRV_IDX] in driver_idxs.keys():
        result[driver_offset + driver_idxs[row[DRV_IDX]]] = 1
    if row[BUS_IDX] in buses_idxs.keys():
        result[buses_offset + buses_idxs[row[BUS_IDX]]] = 1

    date = lpputils.parsedate(row[DEP_IDX]).date()

    holiday = 0
    school_hol = 0
    summer_hol = 0

    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[-3] = summer_hol
    result[-2] = holiday
    result[-1] = school_hol

    return result


def model4(row):
    """
    MODEL4 : binary day and week attributes + all holiday (binary)
    indeksi : 0-6 dnevi, 7-31 ura, pocitnice 3x
    server: 150.86627
    lokalno: 128.68312
    """
    result = np.zeros(7 + 24 + 3)

    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    result[day] = 1
    result[7 + hour] = 1

    date = lpputils.parsedate(row[DEP_IDX]).date()

    holiday = 0
    school_hol = 0
    summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[-3] = summer_hol
    result[-2] = holiday
    result[-1] = school_hol

    return result


def model5(row):
    """
    MODEL5 : binary day and week attributes + all holiday (binary)
    indeksi : 0-6 dnevi, 7-31 ura, pocitnice 3x, padavine
    server: 184.51330
    lokalno:  147.68
    """
    global arso
    result = np.zeros(7 + 24 + 4)

    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    result[day] = 1
    result[7 + hour] = 1

    date = lpputils.parsedate(row[DEP_IDX]).date()

    holiday = 0
    school_hol = 0
    summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[-4] = summer_hol
    result[-3] = holiday
    result[-2] = school_hol

    if date.strftime("%Y-%m-%d") in arso.keys():
        result[-1] = arso[date.strftime("%Y-%m-%d")][0]
    else:
        print("{0} - No weather data !!!".format(date.strftime("%Y-%m-%d")))

    return result


def model6(row):
    """
    MODEL6 : binary day and week attributes + all holiday (binary) + rush
    indeksi : 0-6 dnevi, 7-31 ura, pocitnice 3x, rush1, rush2
    server: ??
    lokalno: 147.68
    """
    result = np.zeros(7 + 24 + 5)

    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    result[day] = 1
    result[7 + hour] = 1

    date = lpputils.parsedate(row[DEP_IDX]).date()

    holiday = 0
    school_hol = 0
    summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[-5] = summer_hol
    result[-4] = holiday
    result[-3] = school_hol

    if hour >= 3 and hour <= 6:
        result[-2] = (hour % 3) / 3
    if hour >= 15 and hour <= 18:
        result[-1] = (3 - hour % 15) / 3;

    return result

def model7(row):
    """
    MODEL7 : binary day and week attributes + all holiday (binary) -> added 20 min interval between 06 and 09
    indeksi : 0-6 dnevi, 7-37 ura, pocitnice 3x, padavine
    server: ?
    lokalno:
    """
    global arso
    result = np.zeros(7 + 30 + 4)

    date = lpputils.parsedate(row[DEP_IDX]).date()
    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    minutes = lpputils.parsedate(row[DEP_IDX]).minute
    result[day] = 1

    if hour < 6:
        result[7 + hour] = 1
    elif 6 <= hour <= 8:
        offset = (hour - 6) * 2
        if 0 <= minutes <= 20:
            result[7 + hour + offset] = 1
        elif 20 < minutes <= 40:
            result[7 + hour + offset + 1] = 1
        elif 40 < minutes <= 59:
            result[7 + hour + offset + 2] = 1
    else:
        result[7 + hour + 6] = 1

    holiday = 0
    school_hol = 0
    summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[-4] = summer_hol
    result[-3] = holiday
    result[-2] = school_hol

    if date.strftime("%Y-%m-%d") in arso.keys():
        result[-1] = arso[date.strftime("%Y-%m-%d")][0]
    else:
        print("No data !!!")

    return result


def model8(row):
    """
    MODEL8 : binary day and week attributes + all holiday (binary) + weather + detour-> added 20 min interval between 06 and 09
    indeksi : 0-6 dnevi, 7-37 ura, pocitnice 3x, padavine, obvoz
    server: ?
    lokalno:
    """
    global arso
    global detours

    result = np.zeros(7 + 30 + 5)

    date = lpputils.parsedate(row[DEP_IDX]).date()
    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    minutes = lpputils.parsedate(row[DEP_IDX]).minute
    result[day] = 1

    if hour < 6:
        result[7 + hour] = 1
    elif 6 <= hour <= 8:
        offset = (hour - 6) * 2
        if 0 <= minutes <= 20:
            result[7 + hour + offset] = 1
        elif 20 < minutes <= 40:
            result[7 + hour + offset + 1] = 1
        elif 40 < minutes <= 59:
            result[7 + hour + offset + 2] = 1
    else:
        result[7 + hour + 6] = 1

    holiday = 0
    school_hol = 0
    summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[-5] = summer_hol
    result[-4] = holiday
    result[-3] = school_hol

    if date.strftime("%Y-%m-%d") in arso.keys():
        result[-2] = arso[date.strftime("%Y-%m-%d")][0]
    else:
        print("No data !!!")

    line = row[2]
    if row[3][0:2] in ['B ', 'G ', 'I ', 'Z ']:
        line += row[3][0]
    result[-1] = check_detour(line, row[DEP_IDX], detours)

    return result

def model9(row):
    """
    MODEL9 : binary day and hour attributes + all holiday (binary) -> added 20 min interval between 06 and 09
    indeksi : 30 * 7 kombinacije dan ura, pocitnice 3x, padavine, detour
    server: 179.68471
    lokalno: 142.....
    uporaba: rezultati 8, 9(+45 sekund na linijo 1), 10(+65 sekund na linijo 1)
    """
    global arso
    result = np.zeros(7*30 + 4)

    date = lpputils.parsedate(row[DEP_IDX]).date()
    day = lpputils.parsedate(row[DEP_IDX]).weekday()
    hour = lpputils.parsedate(row[DEP_IDX]).hour
    minutes = lpputils.parsedate(row[DEP_IDX]).minute

    day_offset = 30 * day

    if hour < 6:
        result[day_offset + hour] = 1
    elif 6 <= hour <= 8:
        offset = (hour - 6) * 2
        if 0 <= minutes <= 20:
            result[day_offset + hour + offset] = 1
        elif 20 < minutes <= 40:
            result[day_offset + hour + offset + 1] = 1
        elif 40 < minutes <= 59:
            result[day_offset + hour + offset + 2] = 1
    else:
        result[day_offset + hour + 6] = 1

    holiday = 0
    school_hol = 0
    summer_hol = 0
    if date in HOLIDAYS:
        holiday = 1
    if date in SCHOOL_HOL:
        school_hol = 1
    if lpputils.parsedate(SUMMER_HOL[0]).date() <= date <= lpputils.parsedate(SUMMER_HOL[1]).date():
        summer_hol = 1

    result[-5] = summer_hol
    result[-4] = holiday
    result[-3] = school_hol

    if date.strftime("%Y-%m-%d") in arso.keys():
        result[-2] = arso[date.strftime("%Y-%m-%d")][0]
    else:
        print("No data !!!")

    line = row[2]
    if row[3][0:2] in ['B ', 'G ', 'I ', 'Z ']:
        line += row[3][0]
    result[-1] = check_detour(line, row[DEP_IDX], detours)

    return result

def model_init(data_train, name):
    """
    Some models need initalization
    """
    global arso
    global detours
    if name in ['MODEL1']:
        driver_average(None, data_train)
        bus_average(None, data_train)
    if name in ['MODEL5', 'MODEL7', 'MODEL8','MODEL9']:
        arso = parse_arso_data()
        max_pad = max(arso.values(), key= lambda x: x[0])[0]
        max_sno = max(arso.values(), key=lambda x: x[1])[1]
        arso = {key: [arso[key][0] / float(max_pad), arso[key][1] / float(max_sno)] for key in arso.keys()}
        detours = parse_detour_data()


def model_getter(name):
    """
    Helper function which returns required model row parser
    """
    if name is 'MODEL1':
        return model1
    elif name is 'MODEL2':
        return model2
    elif name is 'MODEL3':
        return model3
    elif name is 'MODEL4':
        return model4
    elif name is 'MODEL5':
        return model5
    elif name is 'MODEL6':
        return model6
    elif name is 'MODEL7':
        return model7
    elif name is 'MODEL8':
        return model8
    elif name is 'MODEL9':
        return model9