import json
import os
import pickle

ROOT = os.path.dirname(__file__)


def read_data():
    test = r'{}\data\test\all_data_niid_2_keep_0_test_8.json'.format(ROOT)
    train = r'{}\data\train\all_data_niid_2_keep_0_train_8.json'.format(ROOT)
    train = json.load(open(train))
    test = json.load(open(test))
    pickle.dump(train, open(r'{}\data\train\all_data_niid_sf0.2_tf0.8_k0.pkl'.format(ROOT), 'wb'))
    pickle.dump(test, open(r'{}\data\test\all_data_niid_sf0.2_tf0.8_k0.pkl'.format(ROOT), 'wb'))



read_data()