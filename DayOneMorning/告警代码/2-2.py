#!/usr/bin/env python3
import csv
from collections import Counter
from datetime import datetime, timedelta

DATA_DIR = "data/"
OUTPUT_DIR = "output/"
INT_FAULT_NO = "007-103-00-040012"
VALID_STAT = "在网"

def in_this_day(start_time, end_time, day_string):
    """ Decide if the time range falls in the day defined by day_string."""
    max_start_time = datetime.strptime(day_string, "%Y%m%d")
    min_end_time = max_start_time + timedelta(days=1)
    if start_time > max_start_time:
        max_start_time = start_time

    if end_time < min_end_time:
        min_end_time = end_time

    return max_start_time <= min_end_time

def happen_this_day(start_time, day_string):
    """ Decide if start_time falls in the day defined by day_string."""
    day_start = datetime.strptime(day_string, "%Y%m%d")
    day_end = day_start + timedelta(days=1)
    return day_start <= start_time <= day_end

def main():
    """Main function."""
    fault_city = []
    with open(DATA_DIR + "标准化告警.csv", "r", encoding="utf-8-sig") \
        as in_file:
        for line in csv.reader(in_file):
            start_time = datetime.strptime(line[2],
                                           "%Y/%m/%d %H:%M:%S")
            # end_time = datetime.strptime(line[3],
            #                             "%Y/%m/%d %H:%M:%S")
            # if in_this_day(start_time, end_time, "20180808"):
            if happen_this_day(start_time, "20180808"):
                fault_city.append(line[8])

    counter = Counter(fault_city)
    with open(OUTPUT_DIR + "citys.csv", "w", encoding="utf-8-sig") \
        as out_file:
        print_data = '\n'.join(list(map(str, [x[1] for x in
                                              counter.most_common(3)])))
        out_file.write(print_data)

main()
