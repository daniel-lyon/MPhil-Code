import numpy as np
import pandas as pd
from main import LF
from astropy.cosmology import FlatLambdaCDM

# Read in CDFS main data
data = pd.read_csv('./DATA/ZFOURGE/CDFS/CDFS_MAIN.csv')
df = pd.DataFrame(data)
df = df[df['Use'] == 1]
df = df[df['LIR'] >= 0]
    
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # cosmology
redshift_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)] # redshift bins
lum_bins = np.arange(8, 14, 0.2) # luminosity bins
z = df['zpk'] # redshift
lum = np.log10(df['LIR']) # apparent magnitude

mlim = 27 # apparent magnitude limit
survey_area = 0.03556 # survey area in square degrees

lf = LF(cosmo, z, lum, redshift_bins, lum_bins, mlim, survey_area, min_count=10)
lf.print_counts()
lf.print_volumes()
# lf.plot()
# lf.overlay_plot()
lf.plot_histograms()
lf.plot_volumes()
lf.fit(func='Schechter', verbose=True, maxfev=10000)