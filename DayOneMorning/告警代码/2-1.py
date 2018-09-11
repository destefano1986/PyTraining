#!/usr/bin/env python3
import csv

DATA_DIR = "data/"

def main():
    """Main function."""
    provinces = {}
    with open(DATA_DIR + "省区域.csv", "r", encoding="utf-8-sig") as in_file:
        for _line in csv.reader(in_file):
            province = dict(name=_line[1])
            provinces[_line[0]] = province

    citys = {}
    with open(DATA_DIR + "市区域.csv", "r", encoding="utf-8-sig") as in_file:
        for _line in csv.reader(in_file):
            try:
                _ = provinces[_line[2]]
                city = dict(name=_line[1], belong=_line[2])
                citys[_line[0]] = city
            except KeyError:
                continue

    countys = {}
    with open(DATA_DIR + "县区区域.csv", "r", encoding="utf-8-sig") as in_file:
        for _line in csv.reader(in_file):
            try:
                _ = citys[_line[2]]
                county = dict(name=_line[1], belong=_line[2])
                countys[_line[0]] = county
            except KeyError:
                continue

    sites_by_name = {}
    with open(DATA_DIR + "基站.csv", "r", encoding="utf-8-sig") as in_file:
        for _line in csv.reader(in_file):
            try:
                _ = countys[_line[3]]
                site = dict(site_id=_line[0], belong=_line[3],
                            stat=_line[4])
                sites_by_name[_line[1]] = site
            except KeyError:
                continue

    faults = []
    with open(DATA_DIR + "告警.csv", "r", encoding="utf-8-sig") as in_file:
        for _line in csv.reader(in_file):
            try:
                site = sites_by_name[_line[4]]
                site_id = site['site_id']
                stat = site['stat']
                county = site['belong']
                city = countys[county]['belong']
                province = citys[city]['belong']

            except KeyError:
                continue

            fault = (_line[0], _line[1], _line[2], _line[3], _line[4],
                     site_id, stat, provinces[province]['name'],
                     citys[city]['name'], countys[county]['name'])
            faults.append(fault)

    with open(DATA_DIR + "标准化告警.csv", "w", encoding="utf-8-sig") \
        as in_file:
        csv_writer = csv.writer(in_file)
        csv_writer.writerows(faults)

main()
