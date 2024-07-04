import pandas as pd
import pandas_market_calendars as mcal

from functools import wraps
from datetime import datetime
from typing import Callable
from enum import Enum, unique


def validate_dates(func: Callable) -> Callable:
    """
    <DESCRIPTION>
    Decorator to test whether start_date is earlier than end date.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> pd.DataFrame:
        start_date: str = kwargs.get('start_date')
        end_date: str = kwargs.get('end_date')
        start_dt: datetime = datetime.strptime(start_date,
                                               '%Y-%m-%d')
        end_dt: datetime = datetime.strptime(end_date,
                                             '%Y-%m-%d')
        if start_dt >= end_dt:
            raise ValueError("Start date must be earlier than end date.")
        return func(*args, **kwargs)
    return wrapper


@unique
class Market(Enum):
    KRX = "XKRX"
    NYSE = "NYSE"

    @classmethod
    def list_members(cls) -> str:
        return ', '.join(cls.__members__.keys())


class BusinessDates:
    """
    <DESCRIPTION>
    Print business dates for each markets.

    <PARAMS>
    market: Market to consider.
    """

    def __init__(self,
                 market: Market):
        self.market: Market = market
        self.calendar = self._get_calendar()

        self.calendar.remove_time('break_start')
        self.calendar.remove_time('break_end')

    def _get_calendar(self) -> mcal.MarketCalendar:
        return mcal.get_calendar(self.market.value)

    @validate_dates
    def get_schedule(self,
                     start_date: str,
                     end_date: str) -> pd.DataFrame:
        schedule = self.calendar.schedule(start_date=start_date,
                                          end_date=end_date)
        trading_days = schedule.index
        return pd.DataFrame(trading_days, columns=['Date'])


if __name__ == "__main__":
    krx_dates = BusinessDates(Market.KRX)
    schedule = krx_dates.get_schedule(start_date='2000-01-01',
                                      end_date='2024-06-30')
    print(schedule.head())
