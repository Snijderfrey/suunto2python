#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd


class exercise_data:
    """
    Imports data recorded by a Suunto Ambit 3 Peak into a Pandas DataFrame.

    Currently, the data is imported best from the json file found in the Suunto
    App folder. Generally, an import of json files exported from
    quantified-self.io is also possible, however implemented only very
    rudimentary (might be expanded in the future). The data is stored in the
    Pandas DataDrame self.exercise_data.
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
        if ('baro', 'Speed') in self.exercise_data.columns:
            self.exercise_data[('gps', 'Pace')] = 1/self.exercise_data[
                ('baro', 'Speed')]*1000/60
        if ('baro', 'Cadence') in self.exercise_data.columns:
            self.exercise_data[('baro', 'Cadence')] *= 60

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

        import_modes = ['suunto_json', 'qs_json']

        if self.mode == import_modes[0]:  # 'suunto_json'
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

            self.exercise_data = baro
            for ii in [gps, ibi]:
                if len(ii) > 0:
                    self.exercise_data = self.exercise_data.join(ii)

            self.exercise_data = self.exercise_data[
                ~self.exercise_data.index.duplicated(keep='first')]

            self.unparsed_lines = len(self.exercise_raw_data) - len(
                processed_samples)
            self.processed_samples_mask = np.ones_like(
                self.exercise_raw_data, dtype=bool)
            self.processed_samples_mask[processed_samples] = False
            self.unparsed_data = self.exercise_raw_data[
                self.processed_samples_mask]

        elif self.mode == import_modes[1]:  # 'qs_json'
            # currently very rudimentary
            self.exercise_raw_data = json.loads(self.exercise_raw_data)
            self.ibi_values = np.array(
                self.exercise_raw_data['activities'][0]['streams'][6]['data'])

        else:
            raise ValueError(
                'No valid mode entered. Allowed modes are {}'.format(
                    import_modes))
