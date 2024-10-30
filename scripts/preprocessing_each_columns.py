import logging
import pandas
import os ,sys
sys.path.append(os.path.abspath('../data'))

from locationMapping import location_mapping


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def processing_location_col(df):
    logger.info("preprocessing the location column")
    try:
        df['Location'] = df['Location'].replace(location_mapping)
        return df
    except Exception as e:
        logger.error(f"error occured while preprocessing the location column")
        return None
