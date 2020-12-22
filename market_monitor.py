import datetime
import enum
import json
import logging
import os
from pathlib import Path
import string
import sys
import time
import typing
from dotenv import load_dotenv
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests


LOGGING_FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


LOGS_DIR = Path('.') / 'logs'


def mkdir_logs():
    try:
        os.mkdir(LOGS_DIR)
    except FileExistsError:
        pass


def get_now_timestamp_string():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')


@enum.unique
class Role(enum.Flag):
    UNKNOWN = 0
    SUPPORT = enum.auto()
    RESISTANCE = enum.auto()
    BOTH = SUPPORT | RESISTANCE


class Level(typing.NamedTuple):
    price: float
    weight: int
    role: Role
    starts_from: pd.Timestamp

    def __repr__(self) -> str:
        return f'<L {self.price},w={self.weight}({self.role})>'

    def __lt__(self, rhs):
        return self.price < rhs.price


class Zone:
    def __init__(self, level: Level = None, *, low: float = 0, high: float = 0, weight: int = 0, role: Role = Role.UNKNOWN):
        '''
        low: Lower bound of this zone
        high: Higher bound of this zone
        weight: How strong this zone is
        role: The role of this zone
        '''
        if level is None:
            self.low = low
            self.high = high
            self.weight = weight
            self.role = role
        else:
            self.low = level.price
            self.high = level.price
            self.weight = level.weight
            self.role = level.role

    def __repr__(self) -> str:
        return f'<Z {self.low:.5f}~{self.high:.5f},w={self.weight}({self.role})>'

    def is_support(self) -> bool:
        return self.role & Role.SUPPORT == Role.SUPPORT

    def is_resistance(self) -> bool:
        return self.role & Role.RESISTANCE == Role.RESISTANCE


class ZoneFinder:
    @classmethod
    def find_levels(cls, df, threshold=3):
        levels = []
        highs = df['High']
        lows = df['Low']
        length = len(highs)

        for i in range(length):
            if i < threshold or i > length - threshold:
                continue

            timestamp = pd.Timestamp(df.index[i])

            maxima = highs[i]
            score = 1
            while i - score >= 0 and i + score < length:
                if highs[i - score] < maxima > highs[i + score]:
                    score += 1
                else:
                    break
            if score > threshold:
                levels.append(Level(maxima, score, Role.RESISTANCE, timestamp))

            minima = lows[i]
            score = 1
            while i - score >= 0 and i + score < length:
                if lows[i - score] > minima < lows[i + score]:
                    score += 1
                else:
                    break
            if score > threshold:
                levels.append(Level(minima, score, Role.SUPPORT, timestamp))

        levels.sort()
        return levels

    @classmethod
    def merge_levels_into_zones(cls, levels, merge_threshold=0.0004, zone_weight_threshold=5):
        if not levels:
            return []

        # TODO: better merging and scoring zones
        # TODO: The more recent, the more important
        # TODO: add Open price if possible
        zones = []

        # Merge from the lowest price level to the highest
        zone = Zone(levels[0])
        for level in levels[1:]:
            if (level.price - zone.high) / level.price < merge_threshold:
                zone.high = level.price
                zone.weight += level.weight
                zone.role |= level.role
            else:
                zones.append(zone)
                zone = Zone(level)
        zones.append(zone)

        zones = [zone for zone in zones if zone.weight > zone_weight_threshold]
        return zones

    @classmethod
    def expand_zones_in_place(cls, zones, ratio=0.00005):
        for zone in zones:
            zone.high += zone.high * ratio
            zone.low -= zone.low * ratio

    @classmethod
    def find_key_zones_and_levels(cls, df, *, find_levels_threshold=3, merge_levels_threshold=0.0004, zone_weight_threshold=5):
        levels = cls.find_levels(df, find_levels_threshold)
        zones = cls.merge_levels_into_zones(
            levels, merge_levels_threshold, zone_weight_threshold)
        cls.expand_zones_in_place(zones)
        return (zones, levels)


class SetupFinder:
    @ staticmethod
    def current_setup(zones: list[Zone], df_minor_timeframe_latest: pd.DataFrame) -> str:

        def is_bar_intersects_with_zone(bar, zone):
            if bar.Low > zone.high:
                return False
            if bar.High < zone.low:
                return False
            return True

        last_bar = df_minor_timeframe_latest.tail(1)
        last_open = last_bar.Open[0]
        last_high = last_bar.High[0]
        last_low = last_bar.Low[0]
        last_close = last_bar.Close[0]

        second_last_bar = df_minor_timeframe_latest.tail(2).head(1)
        second_last_high = second_last_bar.High[0]
        second_last_low = second_last_bar.Low[0]

        third_from_last_bar = df_minor_timeframe_latest.tail(3).head(1)
        third_from_last_high = third_from_last_bar.High[0]
        third_from_last_low = third_from_last_bar.Low[0]

        is_ascending = second_last_high < last_high and second_last_low < last_low
        is_descending = second_last_high > last_high and second_last_low > last_low

        for zone in zones:
            if zone.is_support():
                if last_low < zone.low < last_high:
                    if is_ascending:
                        if max(last_open, last_close) < (last_high + last_low) / 2:
                            return 'Pinbar on Support'
            if zone.is_resistance():
                if last_high > zone.high > last_low:
                    if is_descending:
                        if min(last_open, last_close) > (last_high + last_low) / 2:
                            return 'Pinbar on Resistance'

            for _, bar in df_minor_timeframe_latest.iterrows():
                if is_bar_intersects_with_zone(bar, zone):
                    if zone.is_support() and is_descending:
                        if third_from_last_low < zone.low > last_low:
                            if third_from_last_high <= second_last_high and third_from_last_low <= second_last_low:
                                return 'Tick down on Support'
                    if zone.is_resistance() and is_ascending:
                        if third_from_last_high > zone.high < last_high:
                            if third_from_last_high >= second_last_high and third_from_last_low >= second_last_low:
                                return 'Tick up on Resistance'
                    break
        return ''


class ChartPlotter:
    LEVEL_COLOR = {
        Role.SUPPORT: 'g',
        Role.RESISTANCE: 'r',
    }
    ZONE_COLOR = {
        Role.UNKNOWN: 'k',
        Role.SUPPORT: 'g',
        Role.RESISTANCE: 'r',
        Role.BOTH: 'xkcd:gold',
    }

    @classmethod
    def get_levels_plot_config(cls, levels):
        def normalized_level_linewidth(weight):
            return (weight + 9) // 10

        # TODO: Wait for multiple fill_between() implementation:
        # https://github.com/matplotlib/mplfinance/issues/292
        hlines = dict(hlines=[], linewidths=[], colors=[], alpha=0.1)
        for level in levels:
            hlines['hlines'].append(level.price)
            hlines['linewidths'].append(
                normalized_level_linewidth(level.weight))
            hlines['colors'].append(cls.LEVEL_COLOR[level.role])
        return hlines

    @classmethod
    def plot(cls, df_in_major_timeframe, major_timeframe_in_minutes, df_in_minor_timeframe, minor_timeframe_in_minutes, zones=[], levels=[], title='', fig=None, ax_major_timeframe=None, ax_minor_timeframe=None):
        if fig is None:
            fig = mpf.figure(figsize=(10, 5))
            fig.suptitle(title)
            plt.subplots_adjust(wspace=0, hspace=0)
            major_timeframe_style = mpf.make_mpf_style()
            minor_timeframe_style = mpf.make_mpf_style(y_on_right=True)
            ax_major_timeframe = fig.add_subplot(
                1, 2, 1, style=major_timeframe_style)
            ax_minor_timeframe = fig.add_subplot(
                1, 2, 2, style=minor_timeframe_style, sharey=ax_major_timeframe)

        major_timeframe_bar_nums = len(df_in_major_timeframe.index)
        minor_timeframe_bar_nums = len(df_in_minor_timeframe.index)
        for zone in zones:
            kwargs = dict(y1=zone.low, y2=zone.high,
                          color=cls.ZONE_COLOR[zone.role], alpha=0.4)
            ax_major_timeframe.fill_between(
                x=(0, major_timeframe_bar_nums), **kwargs)
            ax_minor_timeframe.fill_between(
                x=(0, minor_timeframe_bar_nums), **kwargs)

        # TODO: draw vertical lines adapted to major timeframe
        # currently only draw hourly separator
        timestamps_on_the_hour = [t for t in map(
            pd.Timestamp, df_in_minor_timeframe.index.values) if t.minute == 0]
        vlines = dict(vlines=timestamps_on_the_hour, colors='c',
                      linestyle='-.', linewidths=2, alpha=0.4)
        # To draw levels or not
        #hlines = cls.get_levels_plot_config(levels)
        hlines = []

        kwargs = dict(type='candle', ylabel='', xrotation=15, hlines=hlines)
        mpf.plot(df_in_major_timeframe, **kwargs, ax=ax_major_timeframe,
                 axtitle=f'{major_timeframe_in_minutes}K')
        mpf.plot(df_in_minor_timeframe, **kwargs, ax=ax_minor_timeframe,
                 axtitle=f'{minor_timeframe_in_minutes}K', vlines=vlines)

        return (fig, ax_major_timeframe, ax_minor_timeframe)


class RealTimeAPIMock:
    def __init__(self, df, bar_nums_sliding_window=50, major_timeframe_in_minutes=60, minor_timeframe_in_minutes=5):
        self.data_pointer = 0
        self.df = df
        self.df_len = len(self.df)
        self.bar_nums_sliding_window = bar_nums_sliding_window
        self.major_timeframe_in_minutes = major_timeframe_in_minutes
        self.minor_timeframe_in_minutes = minor_timeframe_in_minutes

    def initial_fetch(self):
        if self.data_pointer > 0:
            return
        r1 = self.data_pointer

        self.data_pointer += self.bar_nums_sliding_window * \
            (self.major_timeframe_in_minutes // self.minor_timeframe_in_minutes)
        return self.df.iloc[r1:self.data_pointer, :]

    def fetch_next(self):
        r1 = self.data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.df_len:
            return None
        return self.df.iloc[r1:self.data_pointer, :]


def wait_for_bar_formed(timeframe_in_minutes: int) -> None:
    now = datetime.datetime.now()
    current_seconds = now.minute * 60 + now.second
    period = timeframe_in_minutes * 60
    wait_seconds = period - (current_seconds % period)

    logger.info(f'Wait for {wait_seconds} seconds to get the next bar...')
    time.sleep(wait_seconds + 1)


class DataGrabber:
    def __init__(self, from_symbol, to_symbol, timeframe_in_minutes=15):
        self.from_symbol = from_symbol
        self.to_symbol = to_symbol
        self.timeframe_in_minutes = timeframe_in_minutes
        self.consecutive_grab_data_failed_counter = 0

    def get_symbols(self):
        return f'{self.from_symbol}{self.to_symbol}'

    def get_timeframe(self):
        return self.timeframe_in_minutes

    @classmethod
    def parse_alphavantage_json_string_to_dataframe(cls, json_string):
        results = json.loads(json_string)
        if 'Error Message' in results:
            logger.error('Wrong API result:', json_string)
            return None

        for key in results.keys():
            if key.startswith('Time Series FX '):
                results = results.get(key)
                break
        else:
            logger.error('Cannot find price fields!', json_string)
            return None

        df = pd.DataFrame.from_dict(results).transpose().apply(pd.to_numeric).rename(columns={'1. open': 'Open', '2. high': 'High',
                                                                                              '3. low': 'Low', '4. close': 'Close'})
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Datetime'
        return df.iloc[::-1]

    @classmethod
    def _grab_alphavantage_json_string(cls, from_symbol: str, to_symbol: str, interval: int, outputsize: str):
        '''
        interval: 1 / 5 / 15 / 30 / 60 (minutes)
        outputsize: 'compact', 'full'
        '''
        # https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=EUR&to_symbol=USD&interval=5min&apikey=demo
        api_endpoint = string.Template(
            'https://www.alphavantage.co/query?function=FX_INTRADAY&datatype=json&' +
            'from_symbol=$from_symbol&' +
            'to_symbol=$to_symbol&' +
            'interval=${interval}min&' +
            'outputsize=$outputsize&' +
            'apikey=$apikey').substitute(
            from_symbol=from_symbol,
            to_symbol=to_symbol,
            interval=interval,
            outputsize=outputsize,
            apikey=os.getenv('ALPHAVANTAGE_API_KEY'),
        )
        logger.info(
            f"Grab from: {api_endpoint[:-len('&apikey=XXXXXXXXXXXXXXXX')]} with API key")

        response = requests.get(api_endpoint)
        try:
            if response.status_code == 200:
                return response.text
            else:
                logger.error(
                    f'Unable to grab data from AlphaVantage! Response={response.status_code}/{response.text}')
        except:
            logger.error(f'Unexpected exception:', sys.exc_info()[0])
        return '{}'

    @classmethod
    def grab_alphavantage_dataframe(cls, from_symbol: str, to_symbol: str, interval: int, outputsize: str = 'compact', save_to_filename: str = ''):
        json_string = cls._grab_alphavantage_json_string(
            from_symbol, to_symbol, interval, outputsize)

        if save_to_filename:
            with open(LOGS_DIR / save_to_filename, 'w') as f:
                f.write(json_string)

        return cls.parse_alphavantage_json_string_to_dataframe(json_string)

    def fetch(self):
        filename = f'debug_grab_result_{self.from_symbol}{self.to_symbol}_{self.timeframe_in_minutes}_{get_now_timestamp_string()}.log'
        df = DataGrabber.grab_alphavantage_dataframe(
            self.from_symbol, self.to_symbol, self.timeframe_in_minutes, 'full', filename)

        if df is None:
            logger.info('Grab data failed')
            self.consecutive_grab_data_failed_counter += 1
            if self.consecutive_grab_data_failed_counter >= 3:
                logger.error(
                    f'Grab data failed {self.consecutive_grab_data_failed_counter} times!')
            return None

        self.consecutive_grab_data_failed_counter = 0
        return df


class Monitor:
    BAR_NUMS_SLIDING_WINDOW = 50  # Determine S/R by how many bars?

    @classmethod
    def send_bot_message(cls, message, disable_notification=False):
        SEND_MESSAGE_API_ENDPOINT = f'https://api.telegram.org/bot{os.getenv("TELEGRAM_BOT_TOKEN")}/sendMessage'
        RETRY_LIMIT = 3

        data = {
            'chat_id': os.getenv('TELEGRAM_BOT_SEND_MESSAGE_GROUP_ID'),
            'text': message,
            'disable_notification': disable_notification,
        }
        for _ in range(RETRY_LIMIT):
            response = requests.post(SEND_MESSAGE_API_ENDPOINT, data=data)
            if response.json().get('ok', False):
                return True
            time.sleep(3)
        logger.error(f'Failed to send bot message [{message}]!')
        return False

    @ classmethod
    def resample(cls, df, major_timeframe_in_minutes=60):
        RESAMPLE_MAP = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
        }
        return df.resample(f'{major_timeframe_in_minutes}T').agg(RESAMPLE_MAP).dropna()

    def monitor(cls, data_grabbers, major_timeframe_in_minutes=60, minor_timeframe_in_minutes=5, show_chart=False, save_chart=False):
        while True:
            for grabber in data_grabbers:
                try:
                    df = grabber.fetch()
                    if df is None:
                        continue

                    df_resampled_into_major_timeframe = cls.resample(
                        df).iloc[-cls.BAR_NUMS_SLIDING_WINDOW:]
                    df = df.iloc[-cls.BAR_NUMS_SLIDING_WINDOW:]

                    zones, levels = ZoneFinder.find_key_zones_and_levels(
                        df_resampled_into_major_timeframe, find_levels_threshold=1, merge_levels_threshold=0.0002, zone_weight_threshold=5)

                    market = grabber.get_symbols()
                    df_latest = df.tail(3)
                    if setup := SetupFinder.current_setup(zones, df_latest):
                        message = f'[{get_now_timestamp_string()}] {market} setup: {setup}'
                        if cls.send_bot_message(message):
                            logger.info(f'Sent bot message: `{message}`')
                        else:
                            logger.error(f'Failed to send bot message!')

                    if show_chart or save_chart:
                        plot_config = dict(
                            df_in_major_timeframe=df_resampled_into_major_timeframe,
                            major_timeframe_in_minutes=major_timeframe_in_minutes,
                            df_in_minor_timeframe=df,
                            minor_timeframe_in_minutes=minor_timeframe_in_minutes,
                            zones=zones,
                            levels=levels,
                            title=market,
                        )
                        ChartPlotter.plot(**plot_config)

                        if save_chart:
                            filename = f"fig_{market}_{setup.replace(' ', '-')}_{get_now_timestamp_string()}.jpg"
                            plt.savefig(fname=LOGS_DIR / filename, bbox_inches='tight')
                        if show_chart:
                            mpf.show()
                except:
                    logger.error(f'Unexpected exception:', sys.exc_info()[0])

            try:
                wait_for_bar_formed(minor_timeframe_in_minutes)
            except KeyboardInterrupt:
                logger.info(f'Exit by KeyboardInterrupt')
                exit()

    @ classmethod
    def run_simulation(cls, df_minor_timeframe_full, start_from_frame_num=0, major_timeframe_in_minutes=60, minor_timeframe_in_minutes=5):
        rtapi = RealTimeAPIMock(df_minor_timeframe_full,
                                cls.BAR_NUMS_SLIDING_WINDOW)
        df = rtapi.initial_fetch()

        def step_one_frame():
            nxt = rtapi.fetch_next()
            if nxt is None:
                print('no more data to plot')
                anim.event_source.interval *= 3
                if anim.event_source.interval > 12000:
                    exit()
                return

            nonlocal df
            df = df.append(nxt)

        for _ in range(start_from_frame_num):
            step_one_frame()

        df_resampled_into_major_timeframe = cls.resample(
            df).iloc[-cls.BAR_NUMS_SLIDING_WINDOW:]

        plot_config = dict(
            df_in_major_timeframe=df_resampled_into_major_timeframe,
            major_timeframe_in_minutes=major_timeframe_in_minutes,
            df_in_minor_timeframe=df.iloc[-cls.BAR_NUMS_SLIDING_WINDOW:],
            minor_timeframe_in_minutes=minor_timeframe_in_minutes,
            title='EURUSD',
        )
        fig, ax_major_timeframe, ax_minor_timeframe = ChartPlotter.plot(
            **plot_config)

        pause = False

        def on_click(event):
            nonlocal pause
            pause ^= True
            if pause:
                anim.event_source.stop()
            else:
                anim.event_source.start()

        fig.canvas.mpl_connect('button_press_event', on_click)

        def animate(frame_num):
            print('Frame:', start_from_frame_num + frame_num)
            ax_major_timeframe.clear()
            ax_minor_timeframe.clear()

            step_one_frame()
            df_resampled_into_major_timeframe = cls.resample(
                df).iloc[-cls.BAR_NUMS_SLIDING_WINDOW:]
            zones, levels = ZoneFinder.find_key_zones_and_levels(
                df_resampled_into_major_timeframe, find_levels_threshold=1, merge_levels_threshold=0.0002, zone_weight_threshold=5)

            plot_config = dict(
                df_in_major_timeframe=df_resampled_into_major_timeframe,
                major_timeframe_in_minutes=major_timeframe_in_minutes,
                df_in_minor_timeframe=df.iloc[-cls.BAR_NUMS_SLIDING_WINDOW:],
                minor_timeframe_in_minutes=minor_timeframe_in_minutes,
                zones=zones,
                levels=levels,
                fig=fig,
                ax_major_timeframe=ax_major_timeframe,
                ax_minor_timeframe=ax_minor_timeframe,
            )
            ChartPlotter.plot(**plot_config)
            # plt.pause(0.001)  # reduce GUI freeze

            df_latest = df.tail(3)
            if setup := SetupFinder.current_setup(zones, df_latest):
                print(setup)
                on_click(None)

        anim = animation.FuncAnimation(fig, animate, interval=100)
        mpf.show()


if __name__ == '__main__':
    load_dotenv()
    mkdir_logs()
    # Monitor.run_simulation(DataFeedGrabberTestCase.get_test_dataframe_5min_full())
    data_grabbers = [
        DataGrabber('EUR', 'USD', 15),
        DataGrabber('AUD', 'USD', 15),
        DataGrabber('USD', 'JPY', 15),
        DataGrabber('EUR', 'GBP', 15),
    ]
    Monitor().monitor(data_grabbers, minor_timeframe_in_minutes=15,
                      show_chart=False, save_chart=True)
