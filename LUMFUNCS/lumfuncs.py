import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

class LF(object):
    def __init__(self, cosmo, lum, n_lum_bins, z, z_max, z_bins, survey_area):
        """ 
        Class to calculate the luminosity function of a galaxy sample.
        
        Parameters
        ----------
        cosmo : astropy.cosmology
            Cosmology object.
            
        lum : array-like
            Luminosities of the galaxies.
        
        n_lum_bins : int
            Number of luminosity bins.
            
        z : array-like
            Redshifts of the galaxies.
            
        z_max : array-like
            Maximum redshift of the galaxies.
        
        z_bins : list of tuples
            Redshift bins. Example: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        
        survey_area : float
            Survey area in square degrees.
        """
        self._z = z
        self._z_max = z_max
        self._z_bins = z_bins
        self._lum = lum
        self._n_lum_bins = n_lum_bins
        self._survey_area = survey_area
        df = pd.DataFrame({
            'z': self._z,
            'z_max': self._z_max,
            'lum': self._lum
        })        
        df = df[df['z_max'] > df['z']] # if z_max is less than z, remove the row
        self._df = df
        self._cosmo = cosmo
        
        _, self._lum_bin_edges = np.histogram(self._lum, bins=self._n_lum_bins)
        self._lum_bins = list(zip(self._lum_bin_edges, self._lum_bin_edges[1:]))

    def _bin_data(self):
        """ Bin the data by redshift and luminosity """ 
        all_volumes = []
            
        # Bin the data by redshift
        for min_z, max_z in self._z_bins:
            z_mask = (self._z >= min_z) & (self._z < max_z)
            z_binned_data = self._df[z_mask]
            
            lum = z_binned_data['lum'].values
            z_volumes = []
            
            # Bin the data by luminosity
            for lum_min, lum_max in self._lum_bins:
                lum_mask = (lum >= lum_min) & (lum < lum_max)
                lum_binned_data = z_binned_data[lum_mask]
                
                volumes = self._volume(lum_binned_data, min_z)
                z_volumes.append(volumes)
            
            all_volumes.append(z_volumes)
        
        return all_volumes
                  
    def _volume(self, binned_data, min_z):
        """ Calculate the volume of each source binned by redshift & luminosity """
        dmin = self._cosmo.comoving_distance(min_z).value
        dmaxs = self._cosmo.comoving_distance(binned_data['z']).value
        
        vmin = 4/3 * np.pi * dmin**3
        vmaxs = 4/3 * np.pi * dmaxs**3
        
        volumes = vmaxs - vmin
        
        corrected_area = self._survey_area / 41253
        
        corrected_volumes = volumes * corrected_area
        return corrected_volumes
    
    def phi(self):
        """ Calculate the luminosity function """
        volumes = self._bin_data()
        delta_log_l = np.mean(np.diff(self._lum_bin_edges))
        
        phi_all = []
        for vol_z_bin in volumes:
            phi_z_bin = []
            for vol_lum_bin in vol_z_bin:
                phi_val = 1 / delta_log_l * np.sum(1 / vol_lum_bin)
                phi_z_bin.append(phi_val)
            phi_all.append(phi_z_bin)
        
        return self._lum_bin_edges[:-1], phi_all
    
    def plot(self, min_count=10):
        """ 
        Plot the luminosity function 
        
        Parameters
        ----------
        min_count : int
            Minimum number of galaxies in a luminosity bin required to be plotted.
            Luminosity bins with fewer galaxies will be masked.
        """
        lums, phi = self.phi()
        counts = self.counts(verbose=False)
        for count_z_bin, phi_z_bin, (z_start, z_end) in zip(counts, phi, self._z_bins):
            mask = np.array(count_z_bin) >= min_count
            phi_z_bin = np.array(phi_z_bin)
            plt.plot(lums[mask], phi_z_bin[mask], label=f'{z_start} $\leq$ z < {z_end}')
            plt.xlabel('Luminosity')
            plt.ylabel('Phi')
            plt.yscale('log')
            plt.legend()
            plt.show()
    
    def counts(self, verbose=True):
        """ Get and print the number of galaxies in each luminosity bin """
        volumes = self._bin_data()
        counts = []
        for z_bin in volumes:
            count_z_bin = []
            for lum_bin in z_bin:
                count_lum_bin = len(lum_bin)
                count_z_bin.append(count_lum_bin)
            counts.append(count_z_bin)
        
        if verbose:
            print('Number of galaxies in each luminosity bin:')
            for count, (z_start, z_end) in zip(counts, self._z_bins):
                print(f'{z_start} <= z < {z_end}: {count}. Total = {np.sum(count)}')
            print('\n')
        return counts
    
    def volumes(self, vervose=True):
        """ Get and print the volume of each galaxy in each luminosity bin """
        volumes = self._bin_data()
        total_volumes = []
        for z_bin in volumes:
            volume_z_bin = []
            for lum_bin in z_bin:
                volume_lum_bin = round(np.sum(lum_bin))
                volume_z_bin.append(volume_lum_bin)
            total_volumes.append(volume_z_bin)
        
        if vervose:
            print('Total volume of each luminosity bin:')
            for volume, (z_start, z_end) in zip(total_volumes, self._z_bins):
                vol = [f'{v:.2e}' for v in volume]
                v_total = f'{np.sum(volume):.2e}'
                print(f'{z_start} <= z < {z_end}: {vol}. Total = {v_total}')
            print('\n')
        return total_volumes
         
def fluxtolum(z, fl):
    """Converts flux density [uJy] to  luminosity [erg/s]"""
    # print ' fl is assumed to be in uJy'
    ld = cosmo.luminosity_distance(z)  # ld[Mpc]
    ld *= (3.086 * 10 ** 22)  # ld[m]
    flsi = fl * 1e-32  # [W/m^2/Hz]
    lum = (4 * np.pi * ld ** 2) * flsi / (1 + z) * 1e7  # W -> erg/s
    return lum
        
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from astropy.cosmology import FlatLambdaCDM
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    data = pd.read_csv('./DATA/ZFOURGE/CDFS/CDFS_MAIN.csv')
    df = pd.DataFrame(data) # 30,911 galaxies
    df = df[df['Use'] == 1] # 13,299 galaxies
    df = df[df['FKs'] > 0] # 12,676 galaxies # drop rows if FKs is negative
    df = df[df['FKs'] <= 27] # 11,902 galaxies # drop rows if FKs is negative
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dists = cosmo.comoving_distance(df['zpk']).value # comoving distance
    mag_ab = 25 - 2.5 * np.log10(df['FKs']) # # AB magnitude

    # lum = fluxtolum(df['zpk'], df['FKs'])
    # lum = np.log10(lum)
    
    lum = mag_ab - 5 * np.log10(dists / 10)
    n_lum_bins = 20 # number of luminosity bins
    
    z = df['zpk'] 
    z_max = 0.6 * ((10 ** ((mag_ab - 11.4) / 12.06)) - 2.6) # maximum redshift
    z_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)] # redshift bins
    
    survey_area = 0.11 # square degrees
    
    lf = LF(cosmo, lum, n_lum_bins, z, z_max, z_bins, survey_area)
    lf.counts()
    # lf.volumes()
    lf.plot(min_count=10)