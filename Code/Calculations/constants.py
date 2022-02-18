"""Stores all constants used"""

US_20y_bonds_avg_2_years = [0.0482, 0.0495, 0.0463, 0.0424, 0.0408, 0.0388, 0.0312, 0.0282, 0.0310, 0.0283, 0.0238, 0.0243, 0.0284, 0.0271, 0.0190]
GERM_20y_bonds_avg_2_years = [0.0388, 0.0424, 0.0446, 0.0427, 0.0373, 0.0333, 0.0275, 0.0230, 0.0208, 0.0144, 0.0079, 0.0074, 0.0084, 0.0043, -0.0009]

indexes = ["DAX30", "DJI", "S&P500"]

annual_dates = ["2005-01-01", "2006-01-01", "2007-01-01", "2008-01-01", "2009-01-01", "2010-01-01", "2011-01-01",
                "2012-01-01", "2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01",
                "2019-01-01", "2020-01-01"]

semester_dates = ["2005-01-01", "2005-07-01", "2006-01-01", "2006-07-01", "2007-01-01", "2007-07-01", "2008-01-01",
                  "2008-07-01", "2009-01-01", "2009-07-01", "2010-01-01", "2010-07-01", "2011-01-01", "2011-07-01",
                  "2012-01-01", "2012-07-01", "2013-01-01", "2013-07-01", "2014-01-01", "2014-07-01", "2015-01-01",
                  "2015-07-01", "2016-01-01", "2016-07-01", "2017-01-01", "2017-07-01", "2018-01-01", "2018-07-01",
                  "2019-01-01", "2019-07-01", "2020-01-01", "2020-07-01"]

quarter_dates = ["2005-01-01", "2005-05-01", "2005-09-01", "2006-01-01", "2006-05-01", "2006-09-01", "2007-01-01",
                 "2007-05-01", "2007-09-01", "2008-01-01", "2008-05-01", "2008-09-01", "2009-01-01", "2009-05-01",
                 "2009-09-01", "2010-01-01", "2010-05-01", "2010-09-01", "2011-01-01", "2011-05-01", "2011-09-01",
                 "2012-01-01", "2012-05-01", "2012-09-01", "2013-01-01", "2013-05-01", "2013-09-01", "2014-01-01",
                 "2014-05-01", "2014-09-01", "2015-01-01", "2015-05-01", "2015-09-01", "2016-01-01", "2016-05-01",
                 "2016-09-01", "2017-01-01", "2017-05-01", "2017-09-01", "2018-01-01", "2018-05-01", "2018-09-01",
                 "2019-01-01", "2019-05-01", "2019-09-01", "2020-01-01", "2020-05-01", "2020-09-01"]


trimester_dates = ["2005-01-01", "2005-04-01", "2005-07-01", "2005-10-01", "2006-01-01", "2006-04-01", "2006-07-01",
                   "2006-10-01", "2007-01-01", "2007-04-01", "2007-07-01", "2007-10-01", "2008-01-01", "2008-04-01",
                   "2008-07-01", "2008-10-01", "2009-01-01", "2009-04-01", "2009-07-01", "2009-10-01", "2010-01-01",
                   "2010-04-01", "2010-07-01", "2010-10-01", "2011-01-01", "2011-04-01", "2011-07-01", "2011-10-01",
                   "2012-01-01", "2012-04-01", "2012-07-01", "2012-10-01", "2013-01-01", "2013-04-01", "2013-07-01",
                   "2013-10-01", "2014-01-01", "2014-04-01", "2014-07-01", "2014-10-01", "2015-01-01", "2015-04-01",
                   "2015-07-01", "2015-10-01", "2016-01-01", "2016-04-01", "2016-07-01", "2016-10-01", "2017-01-01",
                   "2017-04-01", "2017-07-01", "2017-10-01", "2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01",
                   "2019-01-01", "2019-04-01", "2019-07-01", "2019-10-01", "2020-01-01", "2020-04-01", "2020-07-01",
                   "2020-10-01"]

annual_window_size = 2
semester_window_size = 4
quarter_window_size = 6
trimester_window_size = 8

annual_months = 12
semester_months = 6
quarter_months = 4
trimester_months = 3

annual = 'annual'
semester = 'semester'
quarter = 'quarter'
trimester = 'trimester'


timeframes_dict = {
    'annual': {
        'time_size_name': annual,
        'timeframe_number': 250,
        'months': annual_months,
        'dates': annual_dates,
        'window_size': annual_window_size
    },
    'semester': {
        'time_size_name': semester,
        'timeframe_number': 125,
        'months': semester_months,
        'dates': semester_dates,
        'window_size': semester_window_size
    },
    'quarter': {
        'time_size_name': quarter,
        'timeframe_number': 84,
        'months': quarter_months,
        'dates': quarter_dates,
        'window_size': quarter_window_size
    },
    'trimester': {
        'time_size_name': trimester,
        'timeframe_number': 63,
        'months': trimester_months,
        'dates': trimester_dates,
        'window_size': trimester_window_size
    }
}

recurrences = ['Auto Recurrence', 'Manual Recurrence']

inputs = ['input 30', 'input 125', 'input 250']

inputs_with_underscore = {
    'input 30': 'input_30',
    'input 125': 'input 125',
    'input 250': 'input 250'
}

num_portfolios = 10000

epochs = 5
batch_size = 5
manual_future_time_steps = 10
