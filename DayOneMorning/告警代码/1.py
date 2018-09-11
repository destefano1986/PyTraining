#!/usr/bin/env python3
import csv

DATA_DIR = "data/"
OUTPUT_DIR = "output/"

def main():
    """Main function."""
    provinces = {}
    with open(DATA_DIR + "省区域.csv", encoding="utf-8-sig") as out_file:
        for _line in csv.reader(out_file):
            province = dict(name=_line[1])
            provinces[_line[0]] = province

    citys = {}
    with open(DATA_DIR + "市区域.csv", encoding="utf-8-sig") as out_file:
        for _line in csv.reader(out_file):
            try:
                _ = provinces[_line[2]]
                city = dict(name=_line[1], belong=_line[2])
                citys[_line[0]] = city
            except KeyError:
                pass

    countys = {}
    with open(DATA_DIR + "县区区域.csv", encoding="utf-8-sig") as out_file:
        for _line in csv.reader(out_file):
            try:
                _ = citys[_line[2]]
                county = dict(name=_line[1], belong=_line[2])
                countys[_line[0]] = county
            except KeyError:
                pass

    sites = {}
    with open(DATA_DIR + "基站.csv", encoding="utf-8-sig") as out_file:
        for _line in csv.reader(out_file):
            try:
                _ = countys[_line[3]]
                is_valid = False
                if _line[4] == "在网":
                    is_valid = True

                site = dict(name=_line[1], belong=_line[3],
                            is_valid=is_valid)
                sites[_line[0]] = site
            except KeyError:
                pass

    with open(OUTPUT_DIR + "count.csv", "w", encoding="utf-8-sig") as out_file:
        print_data = [len(citys), len(countys), len(sites),
                      len([1 for v in sites.values() if v['is_valid']])]
        out_file.write('\n'.join(list(map(str, print_data))))

main()
