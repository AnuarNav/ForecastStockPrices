import os
import time
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta
from Calculations import calculations, constants

curr_date = date(2005, 1,1)
list_dates = []

while curr_date < date(2021,1,1):
    list_dates.append(f""""{curr_date}""")
    curr_date += relativedelta(months=+3)

print('", '.join(map(str,list_dates)))

print(calculations.get_x_months_later_date('2020-08-1', 3))

TotalStartTime = time.time()
print(len(constants.annual_dates))
print(len(constants.semester_dates))
print(len(constants.quarter_dates))
print(len(constants.trimester_dates))

recurrence = 'Auto Recurrence'
input_ = 'input 30'

TotalEndTime = time.time()
total_time_taken_df = pd.DataFrame({'TimeTaken in Minutes': round((TotalEndTime - TotalStartTime) / 60), 'TimeTaken in Hours': round((TotalEndTime - TotalStartTime) / 360)}, index=[0])
total_time_taken_df.to_excel(f'''/Users/anuarnavarro/Desktop/TFG/GitHub/ForecastStockPrices/Code/Data/ModelsMetaResults/Time/{recurrence}/{input_}/input_30_output_250_time.xlsx''')
