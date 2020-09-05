#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class exercise_data:
    '''
    Imports and parses data recorded by a Suunto Ambit 3 Peak into a Pandas
    DataFrame.

    Currently, the data is imported best from the json file found in the Suunto
    App folder. Generally, an import of json files exported from
    quantified-self.io is also possible, however implemented only very
    rudimentary (might be expanded in the future).
    '''

    def __init__(self, json_file, mode='suunto_json'):
        '''
        Initialize instance of exercise_data.

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

        '''
        self.json_file = json_file
        self.mode = mode
        self.parse_json()

    def parse_json(self):
        '''
        Import the json file and parse it into a Pandas DataFrame (currently
        only for self.mode=='suunto_json', for 'qs_json', basically only the
        raw data is imported). The data is stored in self.exercise_data. Some
        calculation on the data are prformed directly after import. Unparsed
        data is stored in self.unparsed_data and can be inspected for possibly
        disregarded data.

        Returns
        -------
        None.

        '''
        with open(self.json_file, 'r') as exercise_file:
            self.exercise_raw_data = exercise_file.read()

        if self.mode == 'suunto_json':  # json file stored by the Suunto
            # Android App. There seems to be a problem with the cadence data.
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

            ibi_1d = pd.Series(
                ibi.stack().values, index=ibi_timeindex.round('S'))
            index_array = np.ones_like(ibi_1d)
            multi_index = pd.MultiIndex.from_arrays(
                [ibi_1d.index, index_array], names=('time', 'data_point'))
            duplicate_indices = multi_index.duplicated(keep='first')

            while True in duplicate_indices:
                index_array += duplicate_indices
                multi_index = pd.MultiIndex.from_arrays(
                    [ibi_1d.index, index_array.astype(int)],
                    names=('time', 'data_point'))
                duplicate_indices = multi_index.duplicated(keep='first')
            ibi_1d.index = multi_index
            ibi_1d = ibi_1d.unstack()
            ibi_1d.columns = pd.MultiIndex.from_product(
                [['IBI'], ibi_1d.columns])
            ibi = ibi_1d

            baro = pd.DataFrame(
                baro_data, index=pd.to_datetime(baro_time).round(freq='S'))
            baro.columns = pd.MultiIndex.from_product([['baro'], baro.columns])
            gps = pd.DataFrame(
                gps_data, index=pd.to_datetime(gps_time).round(freq='S'))
            gps.columns = pd.MultiIndex.from_product([['gps'], gps.columns])

            self.exercise_data = baro.join(gps).join(ibi)
            self.exercise_data = self.exercise_data[
                ~self.exercise_data.index.duplicated(keep='first')]

            # some values are calculated from the raw data
            self.exercise_data[('gps', 'Pace')] = 1/self.exercise_data[
                ('baro', 'Speed')]*1000/60
            self.exercise_data[('heart_rate', 'plain')] = 60000/np.mean(
                self.exercise_data['IBI'], axis=1)

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


if __name__ == "__main__":
    Aug_27_2020 = exercise_data(
        '/home/almami/Alexander/Suunto-Daten/entry_-335030377_1598542618/samples.json')
    Sep_5_2020 = exercise_data(
        '/home/almami/Alexander/Suunto-Daten/entry_1353082632_1599316472/samples.json')

    # Plot of the gps coordinates passed during the exercise ('map').
    plt.figure(0)
    plt.scatter(
        Sep_5_2020.exercise_data[('gps', 'Longitude')],
        Sep_5_2020.exercise_data[('gps', 'Latitude')])

    # Plot of altitude, heart rate and pace over exercise time
    fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax0.plot(Sep_5_2020.exercise_data.index[1:],
             Sep_5_2020.exercise_data[('baro', 'Altitude')][1:])
    ax0.grid(True)
    ax0.set_xlabel('time')
    ax0.set_ylabel('altitude [m]')
    ax1.plot(Sep_5_2020.exercise_data.index[1:],
             Sep_5_2020.exercise_data[('heart_rate', 'plain')][1:])
    ax1.grid(True)
    ax1.set_xlabel('time')
    ax1.set_ylabel('heart rate [1/min]')
    ax2.plot(Sep_5_2020.exercise_data.index[1:],
             Sep_5_2020.exercise_data[('gps', 'Pace')][1:])
    ax2.grid(True)
    ax2.set_xlabel('time')
    ax2.set_ylabel('pace [min/km]')
    ax2.set_ylim(4, 8)
    ax2.set_xlim(
        Sep_5_2020.exercise_data.index[1], Sep_5_2020.exercise_data.index[-1])

    # Plot of IBI values over time
    all_ibis = Sep_5_2020.exercise_data['IBI'].stack()
    ibi_time_values = all_ibis.index.get_level_values(0)

    # plt.figure(2)
    # plt.plot(ibi_time_values[:-1], all_ibis[:-1])

    # Poincaré-Plot of IBI values
    plt.figure(3)
    plt.scatter(all_ibis.values[:-2], np.roll(all_ibis.values[:-1], -1)[:-1])
