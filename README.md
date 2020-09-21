# suunto2python
Import exercise data recorded with Suunto Ambit 3 Peak from Suunto App json
files. 

The data processing uses functions from
https://github.com/Snijderfrey/pyPreprocessing,
so that needs to be available.

The current version is in an early development stage, but was tested
successfully with data obtained with a Suunto Ambit 3 Peak that were
synchronized with Suunto via the Android Suunto App. Data tested so far
was running data (containing GPS, barometer and heart rate data) and
sleep data (containing only barometer and heart rate data).  It might
work as well with other Suunto watches and data, but could not be tested,
yet. 

Json files that can be used for data import were found in the Android data
folder of the Suunto App. 
