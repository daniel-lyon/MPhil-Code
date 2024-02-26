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
redshift_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4,5), (5, 6)] # redshift bins
# redshift_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4,5), (5, 6)] # redshift bins
lum_bins = np.arange(6, 14, 0.25) # luminosity bins
z = df['zpk'] # redshift
lum = np.log10(df['LIR']) # solar luminosities

max_z = 0.652 * (10 ** ((lum - 6.586) / 5.336) - 0.768)

mask = (z <= max_z)
z = z[mask]
lum = lum[mask]

mlim = []
survey_area = 0.03556 # survey area in square degrees

lf = LF(cosmo, z, lum, redshift_bins, lum_bins, mlim, survey_area, min_count=10, ylim=(-6,-2), nrows=3, ncols=2)
# lf.print_counts()
# lf.print_volumes()
# lf.overlay_plot()
func = 'Saunders_lum'
lf.plot_histograms(func)
lf.plot_volumes(func)
# lf.plot()
lf.fit(func, verbose=True, maxfev=100000)