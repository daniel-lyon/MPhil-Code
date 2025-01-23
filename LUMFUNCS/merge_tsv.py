#!/usr/bin/env python3

""" Script for merging ZFOURGE TSV files into a single CSV file """

import pandas as pd
import numpy as np

def read_zf_tsv(filename, skiprows, usecols):
    """ Read in ZFOURGE TSV files and return a dataframe """
    data = pd.read_csv(filename, sep='\t', skiprows=skiprows, usecols=usecols)
    df = pd.DataFrame(data)
    df = df.iloc[2:]
    df['Seq'] = df['Seq'].astype(int)
    df = df.sort_values(by='Seq')
    df = df.set_index('Seq')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

""" CDFS """
# # Get dataframes
# cdfs = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS.tsv', skiprows=375, usecols=['Seq', 'Use', 'RAJ2000', 'DEJ2000'])
# cdfs_sfr = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_SFR.tsv', skiprows=76, usecols=['Seq', 'LIR', 'F24', 'SFR', 'lmass'])
# cdfs_z = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_Z.tsv', skiprows=80, usecols=['Seq', 'zpk'])
# cdfs_flux = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_FLUX.tsv', skiprows=117, usecols=['Seq', 'FU', 'FV', 'FJ'])
# cdfs_xmatch = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_xmatch.tsv', skiprows=54, usecols=['Seq', 'irAGN', 'radAGN', 'xAGN'])
# cdfs_lum_error = pd.read_csv('DATA/ZFOURGE/CDFS/lum_error.csv', index_col=['Seq'], usecols=['Seq', 'l_down', 'l_up'])
# dataframes = [cdfs, cdfs_sfr, cdfs_z, cdfs_flux, cdfs_xmatch, cdfs_lum_error]

# # Export to CSV
# df = pd.concat(dataframes, axis=1)
# cdfs_160_good = np.load('DATA/ZFOURGE/CDFS/red_160.npz', allow_pickle=True)['indexs']
# df['160_good'] = [1 if i in cdfs_160_good else 0 for i in df.index]
# df.to_csv('DATA/ZFOURGE/CDFS/CDFS_MAIN8.csv', sep=',')

""" COSMOS """
# Get dataframes
cosmos = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS.tsv', skiprows=354, usecols=['Seq', 'Use', 'RAJ2000', 'DEJ2000'])
cosmos_sfr = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_SFR.tsv', skiprows=76, usecols=['Seq', 'LIR', 'F24', 'SFR', 'lmass'])
cosmos_z = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_Z.tsv', skiprows=80, usecols=['Seq', 'zpk'])
cosmos_flux = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_FLUX.tsv', skiprows=65, usecols=['Seq', 'FU', 'FV', 'FJ'])
cosmos_xmatch = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_xmatch.tsv', skiprows=54, usecols=['Seq', 'irAGN', 'radAGN', 'xAGN'])
cosmos_lum_error = pd.read_csv('DATA/ZFOURGE/COSMOS/lum_error.csv', index_col=['Seq'])
dataframes = [cosmos, cosmos_sfr, cosmos_z, cosmos_flux, cosmos_xmatch, cosmos_lum_error]

# Export to CSV
df = pd.concat(dataframes, axis=1)
df['160_good'] = [0 for i in df.index]
df.to_csv('DATA/ZFOURGE/COSMOS/COSMOS_MAIN8.csv', sep=',')

""" UDS """
# # Get dataframes
# uds = read_zf_tsv('DATA/ZFOURGE/UDS/UDS.tsv', skiprows=286, usecols=['Seq', 'Use', 'RAJ2000', 'DEJ2000'])
# uds_sfr = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_SFR.tsv', skiprows=76, usecols=['Seq', 'LIR', 'F24', 'SFR', 'lmass'])
# uds_z = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_Z.tsv', skiprows=80, usecols=['Seq', 'zpk'])
# uds_flux = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_FLUX.tsv', skiprows=68, usecols=['Seq', 'FU', 'FV', 'FJ'])
# uds_xmatch = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_xmatch.tsv', skiprows=54, usecols=['Seq', 'irAGN', 'radAGN', 'xAGN'])
# uds_lum_error = pd.read_csv('DATA/ZFOURGE/UDS/lum_error.csv', index_col=['Seq'])
# dataframes = [uds, uds_sfr, uds_z, uds_flux, uds_xmatch, uds_lum_error]

# # Export to CSV
# df = pd.concat(dataframes, axis=1)
# df['160_good'] = [0 for i in df.index]
# df.to_csv('DATA/ZFOURGE/UDS/UDS_MAIN8.csv', sep=',')