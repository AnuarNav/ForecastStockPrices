"""
For every model created (recurrence+input+output), gets the ModelsMetaResults/Time into a df (even if empty) and stores
values calculates for the given recurrence+input+output the following errors: MAPE, MAE, RSME

Saves resulting df into:
/Data/ModelsMetaResults/Time&Errors/{recurrence}/{input}/{time_file_name}_errors.xlsx
"""