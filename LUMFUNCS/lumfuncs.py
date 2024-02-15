import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

    def _bin_volumes(self):
        """ 
        Calculate the volume of each source in each redshift and luminosity bins
        
        Returns
        -------
        all_volumes : 3D array-like
            The volume of each source in each redshift and luminosity bin. (Redshift, Luminosity, Volume)
        """ 
        all_volumes = []
            
        # Bin the data by redshift
        for min_z, max_z in self._z_bins:
            z_mask = (self._z >= min_z) & (self._z < max_z)
            binned_z_data = self._df[z_mask]
            
            binned_z_lums = binned_z_data['lum'].values
            z_volumes = []
            
            # Bin the data by luminosity
            for lum_min, lum_max in self._lum_bins:
                lum_mask = (binned_z_lums >= lum_min) & (binned_z_lums < lum_max)
                binned_lum_data = binned_z_data[lum_mask]
                
                # Calculate the volume of each source in the luminosity bin
                volumes = self._volume(binned_lum_data, min_z, max_z)
                z_volumes.append(volumes)
            
            # Append the volumes of each source in the luminosity bin to the redshift bin
            all_volumes.append(z_volumes)
        
        return all_volumes
                  
    def _volume(self, binned_data, z_bin_min, z_bin_max):
        """ 
        Calculate the volume of each source binned by redshift & luminosity
        
        Parameters
        ----------
        binned_data : pandas.DataFrame
            A set of data binned filtered (binned) by redshift and luminosity.
        
        z_bin_min : float
            Redshift bin minimum. Example: if the redshift bin is (0, 1), z_bin_min = 0.
            
        Returns
        -------
        corrected_volumes : array-like
            The volume of each source in the binned data corrected for the survey area.
        """
        dmin = self._cosmo.comoving_distance(z_bin_min).value
        dmax = self._cosmo.comoving_distance(z_bin_max).value
        
        dmaxs = self._cosmo.comoving_distance(binned_data['z']).value
        
        dmaxs[dmaxs > dmax] = dmax
        
        vmin = 4/3 * np.pi * dmin**3
        vmaxs = 4/3 * np.pi * dmaxs**3
        
        volumes = vmaxs - vmin
        
        if np.any(volumes < 0):
            raise ValueError('Volume of a source is negative. Check the redshifts and the cosmology.')
        
        corrected_area = self._survey_area / 41253 # total area of the sky is 41253 square degrees
        
        corrected_volumes = volumes * corrected_area
        return corrected_volumes
    
    def phi_vmax(self):
        """ 
        Calculate phi values for each redshift and luminosity bin via the 1/Vmax method
        
        Returns
        -------
        lum_bin_edges : array-like
            The edges of the luminosity bins.
        
        phi_all : 2D array-like
            The phi values for each redshift and luminosity bin.
        """
        volumes = self._bin_volumes()
        delta_log_l = np.diff(self._lum_bin_edges)[0]
        phi_all = []
        
        # Volume of each redshift bin
        for vol_z_bin in volumes:
            phi_z_bin = []
            
            # Volume of each luminosity bin
            for vol_lum_bin in vol_z_bin:
                phi_val = (1 / delta_log_l) * np.sum(1 / vol_lum_bin)
                phi_z_bin.append(phi_val)
            phi_all.append(phi_z_bin)
        
        return self._lum_bin_edges[:-1], phi_all
    
    def plot(self, min_count=1):
        """ 
        Plot the luminosity function 
        
        Parameters
        ----------
        min_count : int
            Minimum number of galaxies in a luminosity bin required to be plotted.
            Luminosity bins with fewer galaxies will be masked. Default = 1.
        """
        lums, phi = self.phi_vmax()
        counts = self.counts(verbose=False)
        for count_z_bin, phi_z_bin, (z_start, z_end) in zip(counts, phi, self._z_bins):
            mask = np.array(count_z_bin) >= min_count
            phi_z_bin = np.array(phi_z_bin)
            plt.scatter(lums[mask], phi_z_bin[mask], label=f'{z_start} $\leq$ z < {z_end}')
            plt.plot(lums[mask], phi_z_bin[mask])
            plt.xlabel('Luminosity')
            plt.ylabel('Phi')
            plt.yscale('log')
            plt.legend()
            plt.show()
            
    @staticmethod
    def schechter_magnitude(L, L_star, phi_star, alpha):
        return phi_star * 10 ** (-0.4*(alpha+1)*(L-L_star)) * np.exp(-10.**(-0.4*(L-L_star)))
    
    @staticmethod
    def schechter_luminosity(L, L_star, phi_star, alpha):
        return phi_star * 10 ** (0.4*(alpha+1)*(L-L_star)) * np.exp(-10.**(0.4*(L-L_star)))
        # return phi_star * np.sign(L / L_star) * np.abs(L / L_star) ** (1 - alpha) * np.exp(-L / L_star)
            
    def fit_schechter(self, func, min_count=1, maxfev=1000, verbose=True, show=True):
        """
        Fit the Schechter function to the data. 
        
        Parameters
        ----------
        func : callable
            The function to fit to the data. Options are: 'magnitude', 'luminosity'.
        
        min_count : int
            Minimum number of galaxies in a luminosity bin required to be plotted.
            Luminosity bins with fewer galaxies will be masked. Default = 1.
        
        maxfev : int
            Maximum number of calls to the function. If the optimal parameters are not found, 
            a warning will be raised. Default = 1000.
        
        verbose : bool
            If True, print the optimal parameters of the Schechter function fit. Default = True.
        """
        if func == 'magnitude':
            f = self.schechter_magnitude
        elif func == 'luminosity':
            f = self.schechter_luminosity
        else:
            raise ValueError('func must be "magnitude" or "luminosity"')
        
        replot_data = []
        lums, phi = self.phi_vmax()
        counts = self.counts(verbose=False)
        for count_z_bin, phi_z_bin, (z_start, z_end) in zip(counts, phi, self._z_bins):
            mask = np.array(count_z_bin) >= min_count
            phi_z_bin = np.array(phi_z_bin)
            
            p0 = [self._lum_bin_edges[0], 0.001, -1]
            try:
                params, _ = curve_fit(f, lums[mask], phi_z_bin[mask], p0=p0, maxfev=maxfev)
            except RuntimeError:
                warnings.warn(f'Optimal parameters not found {z_start} <= z < {z_end}: Number of calls to function has reached maxfev = {maxfev}', RuntimeWarning)
                continue
            except TypeError:
                warnings.warn(f'TypeError: The number of func parameters={len(p0)} must not exceed the number of data points={len(lums[mask])}', RuntimeWarning)
                continue    
            
            if verbose:
                print(f'{z_start} <= z < {z_end} Schechter function fit:')
                print(f'L_star = {params[0]:.2e}')
                print(f'phi_star = {params[1]:.2e}')
                print(f'alpha = {params[2]:.2e}')
                print('\n')
                
            replot_data.append([lums[mask], phi_z_bin[mask], f(lums[mask], *params), (z_start, z_end)])
            
            if show:
                plt.scatter(lums[mask], np.log10(phi_z_bin[mask]), label=f'{z_start} $\leq$ z < {z_end}')
                plt.plot(lums[mask], np.log10(f(lums[mask], *params)), label='Schechter fit', color='red', linestyle='--')
                plt.xlabel('$M_{abs}$')
                plt.ylabel('$log(\Phi) (Mpc^{-3})$')
                # plt.yscale('log')
                plt.ylim(-8, -1)
                plt.legend()
            
        return replot_data
    
    def counts(self, verbose=True):
        """ 
        Get and print the number of galaxies in each luminosity bin in each redshift bin
        
        Parameters
        ----------
        verbose : bool
            If True, print the number of galaxies in each luminosity bin in each redshift bin
        
        Returns
        -------
        counts : 2D array-like
            The total number of galaxies in each luminosity bin in each redshift bin
        """
        volumes = self._bin_volumes()
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
        """ 
        Get and print the volume of each luminosity bin in each redshift bin
        
        Parameters
        ----------
        vervose : bool
            If True, print the volume of each luminosity bin in each redshift bin
        
        Returns
        -------
        total_volumes : 2D array-like
            The total volume of each luminosity bin in each redshift bin.
        """
        volumes = self._bin_volumes()
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
    dists = cosmo.luminosity_distance(df['zpk'])  * 10 ** 6 # comoving distance
    mag_ab = 25 - 2.5 * np.log10(df['FKs']) # AB magnitude

    # lum = fluxtolum(df['zpk'], df['FKs'])
    # # lum /= 3.826e33 # convert to solar luminosities
    # lum = np.log10(lum)
    
    lum = mag_ab - 5 * np.log10(dists / 10) # absolute magnitude
    n_lum_bins = 30 # number of luminosity bins
    
    z = df['zpk']
    z_max = 0.6 * ((10 ** ((mag_ab - 11.4) / 12.06)) - 2.6) # maximum redshift
    z_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)] # redshift bins
    
    survey_area = 0.11 # square degrees
    
    lf = LF(cosmo, lum, n_lum_bins, z, z_max, z_bins, survey_area)
    lf.counts()
    # lf.volumes()
    # lf.plot(min_count=0)
    # lf.fit_schechter(func='magnitude', min_count=10, maxfev=1000, show=True)
    
    data = lf.fit_schechter(func='magnitude', min_count=10, maxfev=1000, show=False)
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    for ax, d in zip(axes.flatten(), data):
        x, y, y_fit, (z_start, z_end) = d
        ax.scatter(x, np.log10(y), label=f'{z_start} $\leq$ z < {z_end}')
        ax.plot(x, np.log10(y_fit), label='Schechter fit', color='red', linestyle='--')
        ax.set_xlabel('$M_{abs}$', fontsize=12)
        ax.set_ylabel('$log(\Phi) (Mpc^{-3})$', fontsize=12)
        ax.set_ylim(-8, -1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()
    
    plt.subplots_adjust(hspace=0.3)
    # plt.savefig('schechter_fits.png', dpi=200)
    plt.show()