#!/usr/bin/env python3
import csv
from collections import namedtuple, defaultdict
from datetime import datetime, timedelta

DATA_DIR = "data/"
OUTPUT_DIR = "output/"
INT_FAULT_NO = "007-103-00-040012"
VALID_STAT = "在网"
AUG_8 = datetime(2018, 8, 8)
AUG_9 = datetime(2018, 8, 9)
FAULT = namedtuple('Fault', ['name', 'no'])
FAULT_NAMES = {
    1: FAULT("本地网大面积基站中断一级告警", "007-103-00-940001"),
    2: FAULT("本地网大面积基站中断二级告警", "007-103-00-940002")
}
FAULT_LIMITS = {
    1: 240 * 3600,
    2: 120 * 3600
}

def in_this_day(start_time, end_time, day_string):
    """ Decide if the time range falls in the day defined by day_string."""
    max_start_time = datetime.strptime(day_string, "%Y%m%d")
    min_end_time = max_start_time + timedelta(days=1)
    if start_time > max_start_time:
        max_start_time = start_time

    if end_time < min_end_time:
        min_end_time = end_time

    return max_start_time < min_end_time

def calc_fault_level(current_val):
    """ Calculate fault level."""
    for fault_level in range(1, 3):
        if current_val >= FAULT_LIMITS[fault_level]:
            return fault_level

    return 3

def ceil_minute(current_date):
    """ Ceiling minutes. """
    try:
        if current_date.second > 0:
            current_date += timedelta(seconds=(60 - current_date.second))

        return current_date
    except AttributeError:
        return None


def print_date(current_date):
    """ Convert date to string in format like 2018/08/08 10:20:30. """
    try:
        return datetime.strftime(current_date, "%Y/%m/%d %H:%M:%S")
    except TypeError:
        return ""

def main():
    """Main function."""
    citys = defaultdict(lambda: dict(failed_site_count=0, failed_time_count=0,
                                     current_time=AUG_8, fault_level=3))
    timers = defaultdict(lambda: defaultdict(lambda: dict(
        site=0, duration=0)))
    with open(DATA_DIR + "标准化告警.csv", "r", encoding="utf-8-sig") \
        as in_file:
        for line in csv.reader(in_file):
            start_time = datetime.strptime(line[2],
                                           "%Y/%m/%d %H:%M:%S")
            end_time = datetime.strptime(line[3],
                                         "%Y/%m/%d %H:%M:%S")
            if line[6] == VALID_STAT and line[1] == INT_FAULT_NO \
                    and in_this_day(start_time, end_time, "20180808"):
                city_name = line[8]
                if start_time <= AUG_8:
                    timers[city_name][AUG_8]
                    citys[city_name]['failed_time_count'] += \
                        (AUG_8 - start_time).total_seconds()
                    citys[city_name]['failed_site_count'] += 1
                else:
                    timers[city_name][start_time]['site'] += 1

                if end_time <= AUG_9:
                    timers[city_name][end_time]['site'] -= 1
                    timers[city_name][end_time]['duration'] += \
                        (end_time - start_time).total_seconds()

    faults = []
    for city_name, city_item in timers.items():
        city_item[AUG_9]
        city = citys[city_name]
        Timer = namedtuple("Timer", ["time", "site", "duration"])
        city_timers = [Timer(x[0], x[1]['site'], x[1]['duration'])
                       for x in city_item.items()]
        city_timers.sort(key=lambda x: x.time)
        for timer in city_timers:
            city['failed_time_count'] += city['failed_site_count'] \
                * (timer.time - city['current_time']).total_seconds()
            city['current_time'] = timer.time
            new_fault_level = calc_fault_level(city['failed_time_count'])
            old_fault_level = city['fault_level']
            if new_fault_level < old_fault_level:
                city['fault_level'] = new_fault_level
                for fault_level in range(new_fault_level, old_fault_level):
                    passed_secs = (city['failed_time_count'] \
                        - FAULT_LIMITS[fault_level]) // city['failed_site_count']
                    start_time = max([timer.time - timedelta(seconds=passed_secs), \
                                    AUG_8])

                    faults.append(dict(start_time=start_time,
                                       end_time=None, city_name=city_name,
                                       fault_level=fault_level))

            city['failed_site_count'] += timer.site
            city['failed_time_count'] -= timer.duration
            new_fault_level = calc_fault_level(city['failed_time_count'])
            old_fault_level = city['fault_level']
            if new_fault_level > old_fault_level:
                city['fault_level'] = new_fault_level
                for _, fault in enumerate(faults):
                    if fault['end_time'] is None \
                            and fault['city_name'] == city_name \
                            and fault['fault_level'] < new_fault_level:
                        fault['end_time'] = timer.time

    citys_by_name = {}
    with open(DATA_DIR + "市区域.csv", encoding="utf-8-sig") as in_file:
        for _line in csv.reader(in_file):
            citys_by_name[_line[1]] = _line[0]

    with open(OUTPUT_DIR + "faults.csv", "w", encoding="utf-8-sig") \
        as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerows([[FAULT_NAMES[x['fault_level']].no,
                               print_date(ceil_minute(x['start_time'])),
                               print_date(ceil_minute(x['end_time'])),
                               x['city_name'],
                               FAULT_NAMES[x['fault_level']].name,
                               citys_by_name[x['city_name']]]
                              for x in faults])

main()
