from datetime import date
from dateutil.relativedelta import relativedelta

curr_date = date(2005, 1,1)
list_dates = []

while curr_date <= date(2020,1,1):
    list_dates.append(f""""{curr_date}""")
    curr_date += relativedelta(months=+4)

print('", '.join(map(str,list_dates)))