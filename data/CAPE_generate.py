# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import pickle

import utils.CAPE as cape_utils


def CAPE_parse_raw(raw_dataset_dir, out_dir, subject, cloth_type):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    cape_json = cape_utils.CAPEJson()
    seqs = []
    txt_path = os.path.join(raw_dataset_dir, 'seq_lists', 'seq_list_%s.txt' % subject)
    with open(txt_path, 'r') as f:
        for line in f:
            if cloth_type not in line:
                continue

            seq_name = line.strip().split()[0]
            seq_dir = os.path.join(raw_dataset_dir, 'sequences', subject, seq_name)
            npz_files = sorted(os.listdir(seq_dir))
            pre_idx = None
            frames = []
            for i, npz_file in enumerate(npz_files):
                idx = int(npz_file.strip().split('.')[1])
                if pre_idx is not None and idx != pre_idx + 1:
                    seqs = cape_json.append_seqs(seqs, seq_name, frames)
                    frames = []
                frames = cape_json.append_frames(frames, os.path.join('sequences', subject, seq_name, npz_file))
                pre_idx = idx
            seqs = cape_json.append_seqs(seqs, seq_name, frames)
    cape_json.set_data(subject, cloth_type, seqs)
    cape_json.dump_bin_file(os.path.join(out_dir, '%s_%s.bin' % (subject, cloth_type)))

    print(cape_json.num_of_seqs())
    print(cape_json.num_of_frames())


def split_train_test(out_dir, tag, bin_path, interp_acts, extrap_acts, test_trial):
    def act_in_acts(query_act, acts):
        for act in acts:
            if act in query_act:
                return True
        return False

    cape_json = cape_utils.CAPEJson(bin_path)
    train_list = []
    interp_list = []
    extrap_list = []
    for seq in cape_json.data['seqs']:
        if act_in_acts(seq['seq_name'], extrap_acts):
            extrap_list.append(seq['id'])
        elif act_in_acts(seq['seq_name'], interp_acts):
            if test_trial in seq['seq_name']:
                interp_list.append(seq['id'])
            else:
                train_list.append(seq['id'])
        else:
            train_list.append(seq['id'])

    with open(os.path.join(out_dir, '%s_train.bin' % tag), 'wb') as f:
        pickle.dump(train_list, f)
    with open(os.path.join(out_dir, '%s_interp.bin' % tag), 'wb') as f:
        pickle.dump(interp_list, f)
    with open(os.path.join(out_dir, '%s_extrap.bin' % tag), 'wb') as f:
        pickle.dump(extrap_list, f)

    print(train_list)
    print(interp_list)
    print(extrap_list)


if __name__ == '__main__':
    # CAPE_parse_raw('/mnt/Datasets/CAPE/cape_release', '/mnt/Datasets/CAPE', '03375', 'longlong')
    # split_train_test('/mnt/Datasets/CAPE/', '03375_longlong', '/mnt/Datasets/CAPE/03375_longlong.bin',
    #                  ['box', 'swim', 'twist_tilt'], ['athletics', 'frisbee', 'volleyball'], 'trial1')

    CAPE_parse_raw('/mnt/Datasets/CAPE/cape_release', '/mnt/Datasets/CAPE', '00134', 'shortlong')
