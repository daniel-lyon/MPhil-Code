#!/usr/bin/env python3

""" Script for merging ZFOURGE TSV files into a single CSV file """

import pandas as pd

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
# Get dataframes
cdfs = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS.tsv', skiprows=375, usecols=['Seq', 'Use'])
cdfs_sfr = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_SFR.tsv', skiprows=76, usecols=['Seq', 'LIR'])
cdfs_z = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_Z.tsv', skiprows=80, usecols=['Seq', 'zpk'])
cdfs_flux = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_FLUX.tsv', skiprows=117, usecols=['Seq', 'FU', 'FV', 'FJ'])
cdfs_xmatch = read_zf_tsv('DATA/ZFOURGE/CDFS/CDFS_xmatch.tsv', skiprows=54, usecols=['Seq', 'irAGN', 'radAGN', 'xAGN'])
dataframes = [cdfs, cdfs_sfr, cdfs_z, cdfs_flux, cdfs_xmatch]

# Export to CSV
df = pd.concat(dataframes, axis=1)
df.to_csv('DATA/ZFOURGE/CDFS/CDFS_MAIN2.csv', sep=',')

# """ COSMOS """
# # Get dataframes
# cosmos = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS.tsv', skiprows=354, usecols=['Seq', 'Use'])
# cosmos_sfr = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_SFR.tsv', skiprows=76, usecols=['Seq', 'LIR'])
# cosmos_z = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_Z.tsv', skiprows=80, usecols=['Seq', 'zpk'])
# cosmos_flux = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_FLUX.tsv', skiprows=65, usecols=['Seq', 'FU', 'FV', 'FJ'])
# cosmos_xmatch = read_zf_tsv('DATA/ZFOURGE/COSMOS/COSMOS_xmatch.tsv', skiprows=54, usecols=['Seq', 'irAGN', 'radAGN', 'xAGN'])
# dataframes = [cosmos, cosmos_sfr, cosmos_z, cosmos_flux, cosmos_xmatch]

# # Export to CSV
# df = pd.concat(dataframes, axis=1)
# df.to_csv('DATA/ZFOURGE/COSMOS/COSMOS_MAIN2.csv', sep=',')

# """ UDS """
# # Get dataframes
# uds = read_zf_tsv('DATA/ZFOURGE/UDS/UDS.tsv', skiprows=286, usecols=['Seq', 'Use'])
# uds_sfr = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_SFR.tsv', skiprows=76, usecols=['Seq', 'LIR'])
# uds_z = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_Z.tsv', skiprows=80, usecols=['Seq', 'zpk'])
# uds_flux = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_FLUX.tsv', skiprows=68, usecols=['Seq', 'FU', 'FV', 'FJ'])
# uds_xmatch = read_zf_tsv('DATA/ZFOURGE/UDS/UDS_xmatch.tsv', skiprows=54, usecols=['Seq', 'irAGN', 'radAGN', 'xAGN'])
# dataframes = [uds, uds_sfr, uds_z, uds_flux, uds_xmatch]

# # Export to CSV
# df = pd.concat(dataframes, axis=1)
# df.to_csv('DATA/ZFOURGE/UDS/UDS_MAIN2.csv', sep=',')