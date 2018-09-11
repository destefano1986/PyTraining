#!/usr/bin/env python3
import csv
from datetime import datetime, timedelta
from collections import defaultdict

DATA_DIR = "data/"
OUTPUT_DIR = "output/"
INT_FAULT_NO = "007-103-00-040012"
VALID_STAT = "在网"

def calc_failed_time(start_time, end_time, day_string):
    """ Test the failed time falls in the time range defined by day_string. """
    max_start_time = datetime.strptime(day_string, "%Y%m%d")
    min_end_time = max_start_time + timedelta(days=1)
    if start_time > max_start_time:
        max_start_time = start_time

    if end_time < min_end_time:
        min_end_time = end_time

    return (min_end_time - max_start_time).total_seconds()

def main():
    """Main function."""
    provinces = {}
    with open(DATA_DIR + "省区域.csv", "r", encoding="utf-8-sig") as in_file:
        for line in csv.reader(in_file):
            province = dict(name=line[1])
            provinces[line[0]] = province

    citys = {}
    with open(DATA_DIR + "市区域.csv", "r", encoding="utf-8-sig") as in_file:
        for line in csv.reader(in_file):
            try:
                _ = provinces[line[2]]
                city = dict(name=line[1], belong=line[2])
                citys[line[0]] = city
            except KeyError:
                continue

    countys = {}
    with open(DATA_DIR + "县区区域.csv", "r", encoding="utf-8-sig") as in_file:
        for line in csv.reader(in_file):
            try:
                _ = citys[line[2]]
                county = dict(name=line[1], belong=line[2])
                countys[line[0]] = county
            except KeyError:
                continue

    city_site_data = defaultdict(lambda:
                                 dict(site_count=0, failed_time_count=0))
    with open(DATA_DIR + "基站.csv", "r", encoding="utf-8-sig") as in_file:
        for line in csv.reader(in_file):
            try:
                if line[4] == VALID_STAT:
                    city_name = citys[countys[line[3]]['belong']]['name']
                    city_site_data[city_name]['site_count'] += 1
            except KeyError:
                continue

    with open(DATA_DIR + "标准化告警.csv", "r", encoding="utf-8-sig") \
        as in_file:
        for line in csv.reader(in_file):
            start_time = datetime.strptime(line[2],
                                           "%Y/%m/%d %H:%M:%S")
            end_time = datetime.strptime(line[3],
                                         "%Y/%m/%d %H:%M:%S")
            if line[6] == VALID_STAT and line[1] == INT_FAULT_NO:
                city_name = line[8]
                failed_time = calc_failed_time(start_time, end_time,
                                               "20180808")
                if failed_time > 0:
                    city_site_data[city_name]['failed_time_count'] += failed_time

    min_usable_rate = 1
    min_usable_city = None
    for city_name, city_data in city_site_data.items():
        # 输出格式即为round，故可略过
        usable_rate = 1 - (city_data['failed_time_count'] \
                           / (city_data['site_count'] * 86400))
        if usable_rate < min_usable_rate:
            min_usable_rate = usable_rate
            min_usable_city = city_name

    with open(OUTPUT_DIR + "cityname.csv", "w", encoding="utf-8-sig") \
        as in_file:
        in_file.write("{},{:.2%}".format(min_usable_city, min_usable_rate))

main()
