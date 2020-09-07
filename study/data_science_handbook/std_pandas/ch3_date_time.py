import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from pandas_datareader import data as pd_reader_data

import matplotlib.pyplot as plt
import seaborn

seaborn.set()

date = np.array('2015-07-04', dtype=np.datetime64)

index = pd.DatetimeIndex(['2014-07-04', '2014-08-04', '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)

bday = pd.date_range('2015-07-01', periods=5, freq=BDay())
print("BDay:", bday)

readed_data = pd_reader_data.DataReader("6_Portfolios_2x3", "famafrench")

# 1
# hh = readed_data[9]['SMALL LoBM']
# hh.plot()
# plt.show()
# 2
hh = readed_data[9]['SMALL LoBM']
hh.plot(alpha=0.5, style='-')
hh.resample('M').mean().plot(style=':')
hh.asfreq('M').plot(style='--')
plt.legend(['input', 'resample', 'asfreq'], loc='upper left')
plt.show()


fig, ax = plt.subplots(2, sharex=True)
data = hh.iloc[:10]
data.asfreq('D').plot(ax=ax[0], marker='o')
data.asfreq('D',).plot(ax=ax[1], style='-o')
data.asfreq('D').plot(ax=ax[1], style='--o')
ax[1].legend(["back-fill", "forward-fill"])
plt.show()


fig, ax = plt.subplots(3, sharey=True)
# apply a frequency to the data
goog = hh.asfreq('D')
goog.plot(ax=ax[0])
goog.shift(900).plot(ax=ax[1])
goog.tshift(900).plot(ax=ax[2])
# legends and annotations
local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')
ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=0.3, color='red')
ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=0.3, color='red')
ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=0.3, color='red');
plt.show()