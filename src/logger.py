import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# The above line creates a string for the logfile name using the current date and time.
#The strftime function formats the date and time as a string in the format month_day_year_hour_minute_second.

logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
#This line creates path for the log file. the os.path.join combines current working directory.

os.makedirs(logs_path,exist_ok=True)
#This line creates the directory for the log file if it doesn’t already exist. The exist_ok=True argument means that no error will be raised if the directory already exists.

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

'''
* The above function call configures the logging system. 
* The filename argument specifies the file to write the log messages to.
* The format argument specifies the format of the log messages, which includes the time of the message (%(asctime)s), the line number where the logging call was made (%(lineno)d), the name of the logger (%(name)s), the severity level of the message (%(levelname)s), and the message text (%(message)s). The level argument sets the root logger level to INFO, which means that only events of this severity and above will be tracked.
'''