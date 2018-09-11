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
ALARM = namedtuple('Alarm', ['name', 'no'])
ALARM_NAMES = {
    1: ALARM("本地网大面积断站一级预警", "007-103-00-840001"),
    2: ALARM("本地网大面积断站二级预警", "007-103-00-840002")
}
ALARM_LIMITS = {
    1: 150,
    2: 60
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

def calc_alarm_level(current_val):
    """ Calculate alarm level."""
    for alarm_level in range(1, 3):
        if current_val >= ALARM_LIMITS[alarm_level]:
            return alarm_level

    return 3

def print_date(date):
    """ Print date in format like 2018/08/08 10:20:30. """
    #return datetime.strftime(date, "%Y/%m/%d %H:%M:%S")
    try:
        return datetime.strftime(date, "%Y/%m/%d %T")
    except TypeError:
        return ""

def main():
    """Main function."""
    citys = defaultdict(lambda: dict(failed_site_count=0, alarm_level=3))
    timers = defaultdict(lambda: defaultdict(lambda: 0))
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
                if start_time < AUG_8:
                    timers[city_name][AUG_8]
                    citys[city_name]['failed_site_count'] += 1
                else:
                    timers[city_name][start_time] += 1

                if end_time <= AUG_9:
                    timers[city_name][end_time] -= 1

    alarms = []
    for city_name, city_item in timers.items():
        city = citys[city_name]
        Timer = namedtuple("Timer", ["time", "fail"])
        city_timers = [Timer(x[0], x[1]) for x in city_item.items()]
        city_timers.sort(key=lambda x: x.time)
        for timer in city_timers:
            city['failed_site_count'] += timer.fail
            new_alarm_level = calc_alarm_level(city['failed_site_count'])
            if new_alarm_level == city['alarm_level']:
                continue
            old_alarm_level = city['alarm_level']
            city['alarm_level'] = new_alarm_level
            for alarm_level in range(new_alarm_level, old_alarm_level):
                alarms.append(dict(start_time=timer.time, end_time=None,
                                   city_name=city_name,
                                   alarm_level=alarm_level))
            if old_alarm_level < new_alarm_level:
                for _, alarm in enumerate(alarms):
                    if alarm['city_name'] == city_name and \
                            alarm['alarm_level'] < new_alarm_level and \
                            alarm['end_time'] is None:
                        alarm['end_time'] = timer.time

    citys_by_name = {}
    with open(DATA_DIR + "市区域.csv", encoding="utf-8-sig") as in_file:
        for _line in csv.reader(in_file):
            citys_by_name[_line[1]] = _line[0]

    with open(OUTPUT_DIR + "alarms.csv", "w", encoding="utf-8-sig") \
        as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerows([[ALARM_NAMES[x['alarm_level']].no,
                               print_date(x['start_time']),
                               print_date(x['end_time']), x['city_name'],
                               ALARM_NAMES[x['alarm_level']].name,
                               citys_by_name[x['city_name']]]
                              for x in alarms])

main()
