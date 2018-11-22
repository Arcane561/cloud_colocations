from datetime import datetime, timedelta

def caliop_tai_to_datetime(tai):
    """
    Converts CALIOP profile time in IAT format to a datetime object.

    Arguments:

        tai(float): The CALIOP profile time.

    Returns:

        A datetime object representing the profile time.
    """
    t0 = datetime(1993, 1, 1)
    dt = timedelta(seconds = tai)

    return t0 + dt

