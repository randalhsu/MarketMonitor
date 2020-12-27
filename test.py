import unittest
import dotenv
import pandas as pd
from market_monitor import DataGrabber
import test_data_alphavantage as td


class DataGrabberTestCase(unittest.TestCase):
    COLUMNS = ['Open', 'High', 'Low', 'Close']

    @classmethod
    def get_test_dataframe_5min(cls) -> pd.DataFrame:
        return DataGrabber.parse_alphavantage_json_string_to_dataframe(td.TEST_DATA_ALPHAVANTAGE_EURUSD_5MIN_JSON_STRING)

    def validate_alphavantage_dataframe(self, df: pd.DataFrame) -> None:
        '''
        Example:
                               Open    High     Low   Close
        2020-12-02 09:40:00  1.2058  1.2060  1.2050  1.2052
        2020-12-02 09:45:00  1.2052  1.2058  1.2050  1.2057
        2020-12-02 09:50:00  1.2057  1.2060  1.2054  1.2060
        2020-12-02 09:55:00  1.2059  1.2060  1.2055  1.2058
        2020-12-02 10:00:00  1.2058  1.2059  1.2052  1.2053
        ...                     ...     ...     ...     ...
        2020-12-02 17:35:00  1.2095  1.2097  1.2092  1.2093
        2020-12-02 17:40:00  1.2092  1.2096  1.2092  1.2094
        2020-12-02 17:45:00  1.2094  1.2099  1.2092  1.2098
        2020-12-02 17:50:00  1.2097  1.2099  1.2093  1.2096
        2020-12-02 17:55:00  1.2096  1.2097  1.2093  1.2094

        [100 rows x 4 columns]
        '''
        # check column names and types
        self.assertEqual(type(self).COLUMNS, df.columns.tolist())
        for i in range(len(type(self).COLUMNS)):
            self.assertEqual('float64', str(df.dtypes[i]))
        # check index type
        self.assertEqual('datetime64[ns]', str(df.index.dtype))
        # datetime should be ascending
        self.assertTrue(df.index[0] < df.index[1])

    def test_parse_alphavantage_json_string_to_dataframe(self) -> None:
        data = self.get_test_dataframe_5min()
        self.validate_alphavantage_dataframe(data)

    def test_grab_alphavantage_dataframe(self) -> None:
        data = DataGrabber.grab_alphavantage_dataframe('EUR', 'USD', 5)
        self.validate_alphavantage_dataframe(data)


if __name__ == '__main__':
    dotenv.load_dotenv()
    unittest.main()
