import os
import pandas as pd
import numpy as np

def construct_single_matrix(path, address, output_to):
    df = pd.read_csv(path)
    address = str(address)
    interesting_senders = df[df['from_address'] == address]
    interesting_receivers = df[df['to_address'] == address]
    block_numbers = df['block_number'].unique()

    matrix = pd.DataFrame(0, index=block_numbers, columns=['out_tx', 'in_tx', 'out_value', 'in_value', 'unique_receivers', 'unique_senders']).astype(np.float64)

    for block in block_numbers:
        block_df = df[df['block_number'] == block]
        for i in range(block_df.shape[0]):
            sender = block_df.iloc[i]['from_address']
            receiver = block_df.iloc[i]['to_address']
            value = block_df.iloc[i]['value']
            if sender == address:
                try:
                    matrix.loc[block, 'out_tx'] += 1
                    matrix.loc[block, 'out_value'] += value
                    matrix.loc[block, 'unique_receivers'] += 1
                except:
                    print('Error:', sender, receiver, value)
                    print(matrix.loc[block])
                    continue
            if receiver == address:
                try:
                    matrix.loc[block, 'in_tx'] += 1
                    matrix.loc[block, 'in_value'] += value
                    matrix.loc[block, 'unique_senders'] += 1
                except:
                    print('Error:', sender, receiver, value)
                    print(matrix.loc[block])
                    print(type(value))
                    continue
                

    # Save to csv
    matrix.to_csv(output_to + '/' + address + '.csv')


def construct_matrix(path, blacklist, output_folder='Matrices'):
    df = pd.read_csv(path)
    df_blacklist = pd.read_csv(blacklist)
    blacklist = df_blacklist['ADDRESS']
    
    interesting_senders = df[df['from_address'].isin(blacklist)]
    interesting_receivers = df[df['to_address'].isin(blacklist)]
    block_numbers = df['block_number'].unique()

    interesting_senders.to_csv('interesting_senders.csv')
    interesting_receivers.to_csv('interesting_receivers.csv')

    print('Number of interesting senders:', len(interesting_senders))
    print('Number of interesting receivers:', len(interesting_receivers))
    print('Number of blocks:', len(block_numbers))

    matrix_dict = {}
    for address in interesting_senders['from_address'].unique():
        matrix_dict[address] = pd.DataFrame(0, index=block_numbers, columns=['out_tx', 'in_tx', 'out_value', 'in_value', 'unique_receivers', 'unique_senders']).astype(np.float64)
    for address in interesting_receivers['to_address'].unique():
        matrix_dict[address] = pd.DataFrame(0, index=block_numbers, columns=['out_tx', 'in_tx', 'out_value', 'in_value', 'unique_receivers', 'unique_senders']).astype(np.float64)

    # Construct the matrix
    for block in block_numbers:
        block_df = df[df['block_number'] == block]
        for i in range(block_df.shape[0]):
            sender = block_df.iloc[i]['from_address']
            receiver = block_df.iloc[i]['to_address']
            value = block_df.iloc[i]['value']
            if sender in matrix_dict.keys():
                try:
                    matrix_dict[sender].loc[block, 'out_tx'] += 1
                    matrix_dict[sender].loc[block, 'out_value'] += value
                    matrix_dict[sender].loc[block, 'unique_receivers'] += 1
                except:
                    print('Error:', sender, receiver, value)
                    print(matrix_dict[sender].loc[block])
                    continue
            if receiver in matrix_dict.keys():
                try:
                    matrix_dict[receiver].loc[block, 'in_tx'] += 1
                    matrix_dict[receiver].loc[block, 'in_value'] += value
                    matrix_dict[receiver].loc[block, 'unique_senders'] += 1
                except:
                    print('Error:', sender, receiver, value)
                    print(matrix_dict[receiver].loc[block])
                    print(type(value))
                    continue
                

    # Save to csv
    for key in matrix_dict.keys():
        matrix_dict[key].to_csv(output_folder + '/' + key + '.csv')

import pandas as pd
import numpy as np

def construct_matrix_not_interest(path, blacklist, output_folder='CompareMatrices'):
    df = pd.read_csv(path)
    df_blacklist = pd.read_csv(blacklist)
    blacklist = df_blacklist['ADDRESS']
    
    # Get addresses not in the blacklist
    interesting_senders = df[~df['from_address'].isin(blacklist)]
    interesting_receivers = df[~df['to_address'].isin(blacklist)]
    block_numbers = df['block_number'].unique()


    print('Number of interesting senders:', len(interesting_senders))
    print('Number of interesting receivers:', len(interesting_receivers))
    print('Number of blocks:', len(block_numbers))

    matrix_dict = {}
    for address in interesting_senders['from_address'].unique():
        matrix_dict[address] = pd.DataFrame(0, index=block_numbers, columns=['out_tx', 'in_tx', 'out_value', 'in_value', 'unique_receivers', 'unique_senders']).astype(np.float64)
    for address in interesting_receivers['to_address'].unique():
        matrix_dict[address] = pd.DataFrame(0, index=block_numbers, columns=['out_tx', 'in_tx', 'out_value', 'in_value', 'unique_receivers', 'unique_senders']).astype(np.float64)

    # Construct the matrix
    for block in block_numbers:
        block_df = df[df['block_number'] == block]
        for i in range(block_df.shape[0]):
            sender = block_df.iloc[i]['from_address']
            receiver = block_df.iloc[i]['to_address']
            value = block_df.iloc[i]['value']
            if sender in matrix_dict.keys():
                try:
                    matrix_dict[sender].loc[block, 'out_tx'] += 1
                    matrix_dict[sender].loc[block, 'out_value'] += value
                    matrix_dict[sender].loc[block, 'unique_receivers'] += 1
                except:
                    print('Error:', sender, receiver, value)
                    print(matrix_dict[sender].loc[block])
                    continue
            if receiver in matrix_dict.keys():
                try:
                    matrix_dict[receiver].loc[block, 'in_tx'] += 1
                    matrix_dict[receiver].loc[block, 'in_value'] += value
                    matrix_dict[receiver].loc[block, 'unique_senders'] += 1
                except:
                    print('Error:', sender, receiver, value)
                    print(matrix_dict[receiver].loc[block])
                    print(type(value))
                    continue

    # Save to csv
    for key in matrix_dict.keys():
        matrix_dict[key].to_csv(output_folder + '/' + str(key) + '.csv')