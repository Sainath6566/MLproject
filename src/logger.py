import logging
import os

from datetime import datetime

LOGFILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M')}.log"

log_path = os.path.join(os.getcwd(), "logs", LOGFILE)

os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOGFILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]%(lineno)d%(name)s-%(levelname)s-%(message)s",
    level=logging.INFO
)

