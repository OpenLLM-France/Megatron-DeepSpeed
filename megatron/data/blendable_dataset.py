# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Blendable dataset."""

import hashlib
import os
import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron.core import mpu
from deepspeed.accelerator import get_accelerator

import re

class BlendableDataset(torch.utils.data.Dataset):


    def __init__(self, datasets, weights, size, *,
                 data_cache_path=None):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indices.
        def _build_indices():
            start_time = time.time()
            # see https://github.com/microsoft/Megatron-DeepSpeed/issues/377
            if num_datasets < 255:
                dataset_index = np.zeros(self.size, dtype=np.uint8)
            else:
                raise NotImplementedError('Number of datasets is too large (would require to recompile cpp helpers)')
                dataset_index = np.zeros(self.size, dtype=np.int64)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)

            from megatron.data import helpers
            helpers.build_blending_indices(dataset_index, dataset_sample_index,
                                           weights, num_datasets, self.size,
                                           torch.distributed.get_rank() == 0)
            print_rank_0('> elapsed time for building blendable dataset indices: '
                         '{:.2f} (sec)'.format(time.time() - start_time))
            return dataset_index, dataset_sample_index

        desc = "Blendable dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {weights}\n"
        desc += f"Size: {size}\n"
        self.desc = desc

        if data_cache_path:
            desc_hash = hashlib.md5(desc.encode('utf-8')).hexdigest()
            desc_path = os.path.join(data_cache_path, desc_hash + ".dsc")
            index_path = os.path.join(data_cache_path, desc_hash + "_index.npy")
            sample_index_path = os.path.join(data_cache_path, desc_hash + "_sample_index.npy")
            cache_hit = os.path.isfile(index_path) and os.path.isfile(sample_index_path)
            cache_success = True
            if torch.distributed.get_rank() == 0 and not cache_hit:
                print(' > WARNING: could not find index map files for blendable'
                      ' dataset, building indices on rank 0 ...', flush=True)
                dataset_index, dataset_sample_index = _build_indices()

                # Sort by year, for each dataset where it makes sense
                prefixes = [desc_to_name(dataset.desc) for dataset in datasets]
                groups, groups_meta = group_prefixes(prefixes)
                assert len(groups) == len(groups_meta)
                if len(groups):
                    print("Re-ordering those datasets by year:")
                    for group, meta in zip(groups, groups_meta):
                        assert len(group) == len(meta)
                        for i, (year, prefix) in zip(group, meta):
                            print(f"  {i} ({year}) -- {prefix}")
                        print()
                    enforce_groups_order(groups, dataset_index, dataset_sample_index)

                # Some verbose added (to check number of samples per dataset)
                num_samples_per_dataset = np.bincount(dataset_index)
                assert len(num_samples_per_dataset) == len(datasets)
                counts = {}
                for dataset, num_samples in zip(datasets, num_samples_per_dataset):
                    counts[desc_to_name(dataset.desc)] = num_samples
                for k in sorted(counts):
                    print(f'> dataset {k} -> {counts[k]} samples')

                try:
                    os.makedirs(os.path.dirname(index_path), exist_ok=True)
                    with open(desc_path, 'wt') as fd:
                        fd.write(desc)
                        np.save(index_path, dataset_index, allow_pickle=True)
                        np.save(sample_index_path, dataset_sample_index,
                                allow_pickle=True)
                except OSError:
                    print(f'There was an error trying to create the data cache directory ({data_cache_path})')
                    print('or a file in it. This is set with the --data-cache-path argument. Please')
                    print('ensure you have write access to this directory or specify one that you do have')
                    print('write access to.')
                    cache_success = False

            counts = get_accelerator().LongTensor([cache_success])
            torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
            torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
            if counts[0].item() != (
                torch.distributed.get_world_size() //
                torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()) //
                torch.distributed.get_world_size(group=mpu.get_sequence_parallel_group())):
                print_rank_0("Data index creation unsuccessful, exiting.")
                exit()

            # Load on all ranks.
            print_rank_0(f'> loading blendable dataset index: {index_path}')
            self.dataset_index = np.load(index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_index.size == self.size

            print_rank_0(f'> loading blendable dataset sample index: {sample_index_path}')
            self.dataset_sample_index = np.load(sample_index_path, allow_pickle=True, mmap_mode='r')
            assert self.dataset_sample_index.size == self.size
        else:
            self.dataset_index, self.dataset_sample_index = _build_indices()


        # Check size
        _ = self.__getitem__(self.size - 1)
        try:
            _ = self.__getitem__(self.size)
            raise RuntimeError('BlendedDataset size is improperly bounded')
        except IndexError:
            pass
        print_rank_0('> size of blendable dataset: '
                     '{} samples'.format(self.size))


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return {
            "dataset_idx" : dataset_idx,
            **self.datasets[dataset_idx][sample_idx],
        }

# Helpers added for Training Curriculum

def desc_to_name(desc):
    assert "Data prefix " in desc, desc
    return desc.split("Data prefix ")[1].split("\n")[0]


def get_year(prefix):
    year = re.search(r"\d{4}", prefix)
    if year:
        year = year.group()
        name = re.sub(year, "XXXX", prefix)
    else:
        year = None
        name = prefix
    return name, year


def group_prefixes(prefixes):
    groups = {}
    for i, prefix in enumerate(prefixes):
        name, year = get_year(prefix)
        if year:
            if name not in groups:
                groups[name] = []
            groups[name].append((i, (year, prefix)))
    # Sort by year
    for name, group in groups.items():
        groups[name] = sorted(group, key=lambda x: x[1])
    groups_indices = [[g[0] for g in group] for group in groups.values() if len(group) > 1]
    groups_meta = [[g[1] for g in group] for group in groups.values() if len(group) > 1]
    return groups_indices, groups_meta


def enforce_groups_order(groups, dataset_index, dataset_sample_index):
     for group in groups:
        enforce_group_order(group, dataset_index, dataset_sample_index)


def enforce_group_order(group, dataset_index, dataset_sample_index=None):
    """
    Re-order the integers in an array to ensure that some of them appear in a given order.

    Args:
        group: list
            A list of integers, which we want to appear in given order in "dataset_index"
        dataset_index: np.array
            An array of integers
        dataset_sample_index: np.array
            An array of integers, which we want to reorder accordingly.
            Caution : some assumptions are made here (for a given dataset, samples will remain in crescent order)
    """

    # First find the indices corresponding to each integer in the list
    all_indices = []
    for dataset_idx in group:
        all_indices.append(
            np.where(dataset_index == dataset_idx)[0]
        )
    for i, (dataset_idx, indices) in enumerate(zip(group, all_indices)):
        other_indices = [all_indices[j] for j in range(i+1, len(all_indices))]
        reorder = True
        if not other_indices:
            reorder = False
        else:
            other_indices = np.concatenate(other_indices)
            if not other_indices.size:
                reorder = False
            else:
                other_indices = np.sort(other_indices)

        if reorder:
            min_indice_other, i_min_indice_other = other_indices[0], 0

            max_indice, i_max_indice = indices[-1], len(indices) - 1

            while max_indice > min_indice_other:
                # Swap the two indices
                dataset_index[max_indice], dataset_index[min_indice_other] = dataset_index[min_indice_other], dataset_index[max_indice]
                if dataset_sample_index is not None:
                    dataset_sample_index[max_indice], dataset_sample_index[min_indice_other] = dataset_sample_index[min_indice_other], dataset_sample_index[max_indice]

                # Update
                i_min_indice_other += 1
                if i_min_indice_other >= len(other_indices):
                    break
                min_indice_other = other_indices[i_min_indice_other]
                i_max_indice -= 1
                if i_max_indice < 0:
                    break
                max_indice = indices[i_max_indice]

            # Update the other indices
            for j, idx in enumerate(group[i+1:]):
                all_indices[i+j+1] = np.where(dataset_index == idx)[0]

        if dataset_sample_index is not None:
            # Re-order
            indices = np.where(dataset_index == dataset_idx)[0]
            dataset_sample_index[indices] = np.sort(dataset_sample_index[indices])