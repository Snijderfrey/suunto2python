#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

from pyPreprocessing.smoothing import smoothing, filtering


class exercise_data:
    """
    Imports data recorded by a Suunto Ambit 3 Peak into a Pandas DataFrame.

    Currently, the data is imported best from the json file found in the Suunto
    App folder. Generally, an import of json files exported from
    quantified-self.io is also possible, however implemented only very
    rudimentary (might be expanded in the future).
    """

    def __init__(self, json_file, mode='suunto_json'):
        """
        Initialize instance of exercise_data.

        Some calculation on the data are performed directly after import.

        Parameters
        ----------
        json_file : str
            Path to the file to be imported.
        mode : str, optional
            Gives the file origin of the imported json file. Allowed values are
            'suunto_json' for files from the Suunto App folder and 'qs_json'
            for files exported from quantified-self.io. The default is
            'suunto_json'.

        Returns
        -------
        None.

        """
        self.json_file = json_file
        self.mode = mode
        self.parse_json()

        # Some values are calculated from the raw data.
        # Currently, no arguments are passed to self.filter_ibi() and
        # self.smooth_ibi. Should be added in the future to allow control over
        # filters applied.
        self.ibi_1d_processed = self.ibi_1d.copy()
        self.filter_ibi(maximum=True, minimum=True, lowpass=True,
                        std_factor=2)
        self.smooth_ibi(median=True)
        self.replace_processed_ibi(self.ibi_1d_processed)
        self.exercise_data[('heart_rate', 'raw')] = 60000/np.mean(
            self.exercise_data['IBI_raw'], axis=1)
        self.exercise_data[('heart_rate', 'filtered')] = 60000/np.mean(
            self.exercise_data['IBI_processed'], axis=1)
        if ('baro', 'Speed') in self.exercise_data.columns:
            self.exercise_data[('gps', 'Pace')] = 1/self.exercise_data[
                ('baro', 'Speed')]*1000/60

    def parse_json(self):
        """
        Import the json file and parse it into a Pandas DataFrame.

        (currently only for self.mode=='suunto_json', for 'qs_json', basically
        only the raw data is imported). The data is stored in
        self.exercise_data. Unparsed data is stored in self.unparsed_data and
        can be inspected for possibly disregarded data.

        Returns
        -------
        None.

        """
        with open(self.json_file, 'r') as exercise_file:
            self.exercise_raw_data = exercise_file.read()

        if self.mode == 'suunto_json':  # json file stored by the Suunto
            # Android App.
            self.exercise_raw_data = np.array(
                json.loads(self.exercise_raw_data)['Samples'])

            # interbeat interval (ibi) is collected in lists together with
            # timestamp
            ibi_time = []
            ibi_values = []
            baro_time = []
            baro_data = []
            gps_time = []
            gps_data = []
            processed_samples = []
            for curr_index, curr_sample in enumerate(self.exercise_raw_data):
                if 'R-R' in curr_sample['Attributes']['suunto/sml']:
                    ibi_values.append(
                        curr_sample['Attributes']['suunto/sml']['R-R']['IBI'])
                    ibi_time.append(curr_sample['TimeISO8601'])
                    processed_samples.append(curr_index)
                elif 'Sample' in curr_sample['Attributes']['suunto/sml']:
                    if 'AbsPressure' in curr_sample['Attributes'][
                            'suunto/sml']['Sample']:
                        baro_data.append(
                            curr_sample['Attributes']['suunto/sml']['Sample'])
                        baro_time.append(curr_sample['TimeISO8601'])
                        processed_samples.append(curr_index)
                    if 'Latitude' in curr_sample['Attributes']['suunto/sml'][
                            'Sample']:
                        gps_data.append(
                            curr_sample['Attributes']['suunto/sml']['Sample'])
                        gps_time.append(curr_sample['TimeISO8601'])
                        processed_samples.append(curr_index)

            ibi = pd.DataFrame(ibi_values, index=pd.to_datetime(ibi_time))
            ibi_cumsum = pd.to_timedelta(ibi.stack().cumsum(), unit='ms')
            ibi_timeindex = pd.to_datetime(
                ibi.index[0] -
                pd.Timedelta(ibi.iloc[0].sum(), unit='ms') +
                pd.to_timedelta(ibi_cumsum, unit='ms'))

            self.ibi_1d = pd.Series(
                ibi.stack().values, index=ibi_timeindex.round('S'))

            index_array = np.ones_like(self.ibi_1d)
            multi_index = pd.MultiIndex.from_arrays(
                [self.ibi_1d.index, index_array], names=('time', 'data_point'))
            duplicate_indices = multi_index.duplicated(keep='first')
            while True in duplicate_indices:
                index_array += duplicate_indices
                multi_index = pd.MultiIndex.from_arrays(
                    [self.ibi_1d.index, index_array.astype(int)],
                    names=('time', 'data_point'))
                duplicate_indices = multi_index.duplicated(keep='first')
            ibi = self.ibi_1d
            ibi.index = multi_index
            ibi = ibi.unstack()
            ibi.columns = pd.MultiIndex.from_product(
                [['IBI_raw'], ibi.columns])

            baro = pd.DataFrame(
                baro_data, index=pd.to_datetime(baro_time).round(freq='S'))
            baro.columns = pd.MultiIndex.from_product([['baro'], baro.columns])
            gps = pd.DataFrame(
                gps_data, index=pd.to_datetime(gps_time).round(freq='S'))
            gps.columns = pd.MultiIndex.from_product([['gps'], gps.columns])

            print('gps.shape ', len(gps))
            print('baro.shape ', len(baro))
            print('ibi.shape ', len(ibi))

            self.exercise_data = baro
            for ii in [gps, ibi]:
                if len(ii) > 0:
                    self.exercise_data = self.exercise_data.join(ii)

            self.exercise_data = self.exercise_data[
                ~self.exercise_data.index.duplicated(keep='first')]

            if ('baro', 'Cadence') in self.exercise_data.columns:
                self.exercise_data[('baro', 'Cadence')] *= 60

            self.unparsed_lines = len(self.exercise_raw_data) - len(
                processed_samples)
            self.processed_samples_mask = np.ones_like(
                self.exercise_raw_data, dtype=bool)
            self.processed_samples_mask[processed_samples] = False
            self.unparsed_data = self.exercise_raw_data[
                self.processed_samples_mask]

        elif self.mode == 'qs_json':  # json export from quantified-self.io
            # currently very rudimentary
            self.exercise_raw_data = json.loads(self.exercise_raw_data)
            self.ibi_values = np.array(
                self.exercise_raw_data['activities'][0]['streams'][6]['data'])

    def smooth_ibi(self, median=True, **kwargs):
        """
        Apply smoothing to interbeat intervals (IBIs).

        Parameters
        ----------
        median : bool, optional
            If True, a median filter is applied. The default is True.
        **kwargs : TYPE
            median_window : int
                Is only needed if median is True. Must be an odd number.
                Default is 5.

        Returns
        -------
        None.

        """
        if median:
            median_window = kwargs.get('median_window', 5)
            self.ibi_1d_processed = median_filter(self.ibi_1d_processed,
                                                  size=median_window)
        self.ibi_1d_processed = pd.Series(self.ibi_1d_processed,
                                          index=self.ibi_1d.index)

    def filter_ibi(self, maximum=True, minimum=True, lowpass=True, **kwargs):
        """
        Apply filters to interbeat intervals (IBIs).

        Filters are applied in the order maximum, minimum. The filtered
        data is stored in self.ibi_1d_processed. Filtered values are replaced
        by np.nan.

        Parameters
        ----------
        maximum : bool, optional
            If True, all values above a threshold are removed. The default is
            True.
        minimum : bool, optional
            If True, all values below a threshold are removed. The default is
            True.
        lowpass : bool, optional
            If True, data is filtered based on selective moving average. The
            default is True.
        **kwargs : TYPE
            max_thresh : float
                Is only needed if maximum is True.
            min_thresh : float
                Is only needed if minimum is True.
            weights : list of bool
                Only needed if lowpass is True.
            std_factor : float
                Only needed if lowpass is True.

        Returns
        -------
        None.

        """
        if lowpass:
            weights = kwargs.get('weights', [True, True, False, True, True])
            std_factor = kwargs.get('std_factor', 2)
            self.ibi_1d_processed = np.squeeze(filtering(
                self.ibi_1d_processed.values[np.newaxis], 'spike_filter',
                weights=weights, std_factor=std_factor))
        if maximum:
            maximum_threshold = kwargs.get('max_thresh', 60000/25)
            self.ibi_1d_processed[self.ibi_1d_processed > maximum_threshold] = np.nan
        if minimum:
            minimum_threshold = kwargs.get('min_thresh', 60000/220)
            self.ibi_1d_processed[self.ibi_1d_processed < minimum_threshold] = np.nan

        self.ibi_1d_processed = pd.Series(self.ibi_1d_processed,
                                          index=self.ibi_1d.index)

    def replace_processed_ibi(self, filtered_data):
        """
        Filter IBI data are added to self.exercise_data.

        Parameters
        ----------
        filtered_data : pandas Series
            A pandas Series similar to self.ibi_1d.

        Returns
        -------
        None.

        """
        filtered_data = pd.DataFrame(self.ibi_1d_processed).unstack()
        filtered_data.columns = pd.MultiIndex.from_product(
            [['IBI_processed'], self.exercise_data['IBI_raw'].columns])
        self.exercise_data = self.exercise_data.join(filtered_data)


if __name__ == "__main__":
    Aug_27_2020 = exercise_data(
        '/home/almami/Alexander/Suunto-Daten/entry_-335030377_1598542618/samples.json')
    Sep_5_2020 = exercise_data(
        '/home/almami/Alexander/Suunto-Daten/entry_1353082632_1599316472/samples.json')
    Sep_20_2020 = exercise_data(
        '/home/almami/Alexander/Suunto-Daten/entry_1993441714_1600629134/samples.json')
    Sep_21_2020_sleep = exercise_data(
        '/home/almami/Alexander/Suunto-Daten/entry_1994106508_1600646290/samples.json')

    plot_data = [Sep_20_2020, Sep_21_2020_sleep]

    # # Plot of the gps coordinates passed during the exercise ('map').
    # plt.figure(0)
    # plt.scatter(
    #     plot_data.exercise_data[('gps', 'Longitude')],
    #     plot_data.exercise_data[('gps', 'Latitude')])

    # # Plot of altitude, heart rate and pace over exercise time
    # fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    # ax0.plot(plot_data.exercise_data.index[1:],
    #          plot_data.exercise_data[('baro', 'Altitude')][1:])
    # ax0.grid(True)
    # ax0.set_xlabel('time')
    # ax0.set_ylabel('altitude [m]')
    # ax1.plot(plot_data.exercise_data.index[1:],
    #          plot_data.exercise_data[('heart_rate', 'filtered')][1:])
    # ax1.grid(True)
    # ax1.set_xlabel('time')
    # ax1.set_ylabel('heart rate [1/min]')
    # ax2.plot(plot_data.exercise_data.index[1:],
    #          plot_data.exercise_data[('gps', 'Pace')][1:])
    # ax2.grid(True)
    # ax2.set_xlabel('time')
    # ax2.set_ylabel('pace [min/km]')
    # ax2.set_ylim(4, 8)
    # ax2.set_xlim(
    #     plot_data.exercise_data.index[1], plot_data.exercise_data.index[-1])

    fig_counter = 0
    for curr_data in plot_data:
        # Plot of IBI values over time
        all_ibis = curr_data.exercise_data['IBI_processed'].stack()
        ibi_time_values = np.cumsum(np.diff(all_ibis.index.get_level_values(0)))
        ibi_time_values = np.concatenate(([pd.Timedelta(0)], ibi_time_values))
        ibi_time_values = pd.Series(ibi_time_values)/10**9
    
        plt.figure(fig_counter)
        ax3 = plt.subplot()
        ax3.plot(ibi_time_values[:-1], all_ibis[:-1])
        ax3.set_xlabel('time [s]')
        ax3.set_ylabel('IBI [ms]')
        fig_counter += 1
    
        # Poincaré-Plot of IBI values
        plt.figure(fig_counter)
        ax4 = plt.subplot()
        ax4.scatter(all_ibis.values[:-2], np.roll(all_ibis.values[:-1], -1)[:-1])
        ax4.set_xlabel('IBI(n) [ms]')
        ax4.set_ylabel('IBI(n+1) [ms]')
        fig_counter += 1

