#!/usr/bin/env python3

""" Script for merging ZFOURGE TSV files into a single CSV file """

import pandas as pd

def read_zf_tsv(filename, skiprows):
    """ Read in ZFOURGE TSV files and return a dataframe """
    data = pd.read_csv(filename, sep='\t', skiprows=skiprows)
    df = pd.DataFrame(data)
    df = df.iloc[2:]
    df['Seq'] = df['Seq'].astype(int)
    df = df.sort_values(by='Seq')
    df = df.set_index('Seq')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

""" CDFS """
# Get dataframes
cdfs = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS.tsv', skiprows=375)
cdfs_sfr = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_SFR.tsv', skiprows=76)
cdfs_z = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_Z.tsv', skiprows=80)
cdfs_flux = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_FLUX.tsv', skiprows=117)
dataframes = [cdfs, cdfs_sfr, cdfs_z, cdfs_flux]

# Export to CSV
df = pd.concat(dataframes, axis=1)
df.to_csv('DATA/ZFOURGE/CDFS/CDFS_MAIN.csv', sep=',')

# """ COSMOS """
# # Get dataframes
# cosmos = read_zf_tsv('ZFOURGE/COSMOS.tsv', skiprows=354)
# cosmos_sfr = read_zf_tsv('ZFOURGE/COSMOS_SFR.tsv', skiprows=76)
# cosmos_z = read_zf_tsv('ZFOURGE/COSMOS_Z.tsv', skiprows=80)
# dataframes = [cosmos, cosmos_sfr, cosmos_z]

# # Export to CSV
# df = pd.concat(dataframes, axis=1)
# df.to_csv('ZFOURGE/COSMOS_MAIN.csv', sep=',')

# """ UDS """
# # Get dataframes
# uds = read_zf_tsv('ZFOURGE/UDS.tsv', skiprows=286)
# uds_sfr = read_zf_tsv('ZFOURGE/UDS_SFR.tsv', skiprows=76)
# uds_z = read_zf_tsv('ZFOURGE/UDS_Z.tsv', skiprows=80)
# dataframes = [uds, uds_sfr, uds_z]

# # Export to CSV
# df = pd.concat(dataframes, axis=1)
# df.to_csv('ZFOURGE/UDS_MAIN.csv', sep=',')