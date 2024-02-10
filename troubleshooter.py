
from transaction_reader import get_all_transaction_types_gz
import pickle
import time

# This is used to troubleshoot an issue discovered in picklemakers where it seemed to stop properly impporting data..
# it may be that the protocol changed at some point and the parser broker?

# Running get_all_transaction_types_gz on data_xdd.csv.gz produced:
# {'poc_receipts_v2': 3734595, 'poc_request_v1': 118704, 'poc_receipts_v1': 54818, 'validator_heartbeat_v1': 875682, 'payment_v2': 69544, 'state_channel_open_v1': 3409, 'assert_location_v2': 76207, 'routing_v1': 2322, 'state_channel_close_v1': 3883, 'add_gateway_v1': 34351, 'transfer_hotspot_v2': 3222, 'payment_v1': 6042, 'token_burn_v1': 3930, 'price_oracle_v1': 11566, 'consensus_group_v1': 734, 'rewards_v2': 734, 'consensus_group_failure_v1': 183, 'vars_v1': 12, 'stake_validator_v1': 18, 'unstake_validator_v1': 5, 'transfer_validator_stake_v1': 27, 'assert_location_v1': 7, 'transfer_hotspot_v1': 1, 'oui_v1': 3}
# it seems poc_receipts_v1 stopped appearing after timestamp 1652282983557259163
# Running get_all_transaction_types_gz on data_xde.csv.gz produced:
# {'poc_receipts_v2': 3971032, 'payment_v2': 74438, 'price_oracle_v1': 11769, 'assert_location_v2': 61427, 'validator_heartbeat_v1': 828318, 'payment_v1': 7122, 'routing_v1': 2041, 'add_gateway_v1': 27785, 'token_burn_v1': 1537, 'transfer_hotspot_v2': 5928, 'state_channel_close_v1': 3812, 'state_channel_open_v1': 3084, 'consensus_group_v1': 764, 'rewards_v2': 764, 'unstake_validator_v1': 12, 'consensus_group_failure_v1': 81, 'oui_v1': 4, 'stake_validator_v1': 25, 'assert_location_v1': 1, 'transfer_validator_stake_v1': 55}

# Running get_all_transaction_types_gz on data_xfa.csv.gz produced:  (which means they probably aren't in order..)
# {'poc_request_v1': 649754, 'assert_location_v1': 2473, 'add_gateway_v1': 2093, 'poc_receipts_v1': 334656, 'validator_heartbeat_v1': 12658, 'payment_v1': 2724, 'payment_v2': 3376, 'state_channel_open_v1': 196, 'state_channel_close_v1': 202, 'consensus_group_v1': 272, 'rewards_v1': 244, 'price_oracle_v1': 1861, 'assert_location_v2': 590, 'transfer_hotspot_v1': 185, 'rewards_v2': 28, 'token_burn_v1': 19, 'vars_v1': 1, 'stake_validator_v1': 16, 'routing_v1': 3, 'consensus_group_failure_v1': 1}
transaction_filepath = r'D:\blockchain-etl-export\transactions'

if __name__ == "__main__":
    # file_path = transaction_filepath + "\\" + 'data_xez.csv.gz'
    # print(get_all_transaction_types_gz(file_path,False))

    file_path = transaction_filepath + "\\" + 'all_distances.pickle'
    start_time = time.time()
    print("pickle load start time {}".format(start_time))
    with open(file_path,'rb') as f:
        all_distances = pickle.load(f)
    end_time = time.time()
    print("pickle load end time {} and delta {}".format(end_time,end_time-start_time))

    print(len(all_distances))
    tx_nonzero_count = 0
    for item in all_distances:
        if item[3]:
            tx_nonzero_count +=1
    print(tx_nonzero_count)





