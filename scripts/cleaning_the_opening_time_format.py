import pandas as pd
import re
import datetime as dt
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Function to load data
def load_data(path):
    logger.info("Loading the data")
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return None

# Function to convert time string into 24-hour format
def convert_to_24hr(time_str):
    try:
        # Try to handle time with minutes, e.g. "2:30 PM"
        return dt.datetime.strptime(time_str.strip(), "%I:%M %p").strftime("%H:%M")
    except ValueError:
        try:
            # Handle time without minutes, e.g. "2 PM"
            return dt.datetime.strptime(time_str.strip(), "%I %p").strftime("%H:%M")
        except ValueError:
            try:
                # Handle time without space, e.g. "2PM"
                return dt.datetime.strptime(time_str.strip(), "%I%p").strftime("%H:%M")
            except ValueError:
                return None

# Function to process 'OpeningTimes' column
def process_opening_times(row):
    logger.info("Processing the OpeningTimes column")
    try:
        time_str = row['OpeningTimes']
        # Handle missing or unknown values
        if pd.isnull(time_str) or 'N/A' in time_str or 'check' in time_str.lower():
            return pd.Series(['unknown', 'unknown'])
        # Check if time format is valid (contains '-')
        if '-' in time_str:
            times = time_str.split('-')
            opening_time = convert_to_24hr(times[0])
            closing_time = convert_to_24hr(times[1])
            return pd.Series([opening_time, closing_time])
        # Handle non-standard cases (e.g., "karaoke on Mondays")
        return pd.Series(['Special Schedule', 'Special Schedule'])
    except Exception as e:
        logger.error(f"Error occurred while processing opening times: {e}")
        return pd.Series([None, None])

# Create a new feature: Duration (in hours)
def calculate_duration(row):
    logger.info("Creating a new feature: Duration (in hours)")
    try:
        if row['OpeningTime'] != 'unknown' and row['ClosingTime'] != 'unknown' and \
           row['OpeningTime'] != 'Special Schedule' and row['ClosingTime'] != 'Special Schedule':
            opening_time = dt.datetime.strptime(row['OpeningTime'], '%H:%M')
            closing_time = dt.datetime.strptime(row['ClosingTime'], '%H:%M')
            # Handle overnight duration (closing after midnight)
            if closing_time < opening_time:
                closing_time += dt.timedelta(days=1)
            duration = (closing_time - opening_time).seconds / 3600  # Convert to hours
            return duration
        return None
    except Exception as e:
        logger.error(f"Error occurred while calculating duration: {e}")
        return None

# Function to flag entries with special schedules
def flag_entries_with_special_schedules(df):
    logger.info("Flagging entries with special schedules")
    try:
        df['Is_Special_Schedule'] = df['OpeningTimes'].str.contains('Check|Karaoke|booking required', case=False, na=False)
        return df
    except Exception as e:
        logger.error(f"Error occurred while flagging special schedules: {e}")
        return None

# Final cleaning: Replace NaN or 'Unknown' entries with None for consistency
def final_cleaning(df):
    logger.info("Final cleaning: Replace NaN or 'Unknown' entries with None for consistency")
    try:
        df.replace('unknown', None, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error occurred during final cleaning: {e}")
        return None