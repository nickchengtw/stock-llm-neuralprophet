from datetime import date, timedelta

def generate_date_range(start_date, end_date):
    """
    Generate dates from start_date to end_date, inclusive.
    
    :param start_date: The starting date (datetime.date object).
    :param end_date: The ending date (datetime.date object).
    :yield: Dates between start_date and end_date.
    """
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)
