# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
from typing import Union
from deepspeed import comm as dist
from deepspeed.utils import groups


class CapacityBins(torch.nn.Module):
    """ CapacityBins - maps current capacity value into capacity bins.

        When using drop_tokens=false, the capacity at each iteration will differ since
        we use a capacity to accommodate for the largest number of tokens sent to an expert.
        This creates dynamic shapes tensors.

        The motivation for using bins is to reduce the dynamic shapes to a limited set, hence
        being more friendly when running in non-eager mode (e.g., using compile).

        The minimum range of capacity is the optimal capacity where all tokens are evenly routed
        among all experts. The maximum range of capacity is the worst-case capacity where all
        tokens are routed to a single expert (unlikely, but a valid upper bound).

        This class maintains the current configured capacity bins. It also tracks bins usage info
        which enables to dynamically update the capacity bins to optimize performance (i.e. to
        minimize the number of dummy extra tokens that are routed).

        Upon initialization, before any usage statistics is available, the capacity bins are
        initialized to bins with exponentially growing width.
    """

    def __init__(self,
                 k: int,
                 num_experts: int,
                 num_capacity_bins: int,
                 capacity_bins_exp_base: float,
                 capacity_bins_alignment: int,
                 min_bin_size: int = 1) -> None:
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.num_capacity_bins = num_capacity_bins
        self.capacity_bins_exp_base = capacity_bins_exp_base
        self.configured_alignment = capacity_bins_alignment
        self.min_bin_size = min_bin_size
        assert self.min_bin_size > 0, f'CapacityBins min_bin_size must be > 0, got {min_bin_size}'
        self.device = None
        self.min_tokens_per_expert = None
        self.max_tokens_per_expert = None
        self.alignment = None
        self.capacity_bins = None
        self.bins_usage = None
        self.bins_usage_last = None
        self.bins_usage_since_bins_configured = None

    def set_bins(self, bins: list):
        self.capacity_bins = torch.tensor(bins, dtype=torch.long, device=self.device)
        self.bins_usage = None
        self.bins_usage_last = None

    def get_stats(self, incremental=True):
        if self.bins_usage is None:
            return None

        with torch.no_grad():
            bins_usage = self.bins_usage.clone().detach()
            dist.all_reduce(bins_usage, op=dist.ReduceOp.SUM, group=dist.get_world_group())
            if incremental:
                delta_bins_usage = bins_usage - self.bins_usage_last if self.bins_usage_last is not None else bins_usage
                self.bins_usage_last = bins_usage.clone().detach()
                bins_usage = delta_bins_usage

            bins_usage = bins_usage.to('cpu')
            bins_usage_list = bins_usage.tolist()
            bins_edges = self.capacity_bins.clone().detach().to('cpu')
            bins_edges_list = bins_edges.tolist()
            stats = {
                'min_range': self.min_tokens_per_expert,
                'max_range': self.max_tokens_per_expert,
                'alignment': self.alignment,
                'min_bin_size': self.min_bin_size if self.min_bin_size is not None else 0,
                'edges': bins_edges,
                'usage': bins_usage,
                'summary': {f'bin{i}_{bins_edges_list[i]}': bins_usage_list[i]
                            for i in range(len(bins_usage))}
            }
        return stats

    def get_binned_capacity(self, gate_output, capacity, update_stats=True):
        with torch.no_grad():
            bins = self._get_capacity_bins(gate_output)
            index = torch.searchsorted(bins, capacity, right=False)
            index = torch.min(index, torch.tensor(len(bins) - 1, dtype=capacity.dtype, device=index.device))
            if update_stats:
                self._update_stats(index)
        return bins[index]

    def _update_stats(self, index):
        # currently we maintain stats for training only
        if self.training:
            if self.bins_usage is None:
                self.bins_usage = torch.zeros(self.num_capacity_bins,
                                              dtype=torch.long,
                                              device=index.device,
                                              requires_grad=False).detach()
            self.bins_usage[index] += 1

    def _generate_bins(self, force_start_bin=False):
        # create exponentially growing width bins, and normalize width sum to 1.0
        # when force_start_bin=True, we force the first bin value = start range (aka start).
        # force_start_bin=True is handled by prepending width=0
        start = self.min_tokens_per_expert
        stop = self.max_tokens_per_expert
        exp_base = torch.tensor(self.capacity_bins_exp_base, dtype=torch.float).to(self.device)
        if force_start_bin:
            bin_widths = exp_base**torch.arange(0, self.num_capacity_bins - 1, device=self.device)
            bin_widths = torch.cat([torch.tensor([0.], device=bin_widths.device), bin_widths])
        else:
            bin_widths = exp_base**torch.arange(0, self.num_capacity_bins, device=self.device)
        normalized_bin_widths = bin_widths / torch.sum(bin_widths)

        # calculate bin edges by accumulating the bins width and scaling to [start...stop] range
        # finally, align bin edges
        bin_edges = torch.cumsum(normalized_bin_widths, dim=0)
        bin_edges = start + (stop - start) * bin_edges
        bin_edges = torch.ceil(bin_edges / self.alignment).mul(self.alignment).to(torch.long)

        # verify that we got N distinct capacity bins
        assert len(set(bin_edges.tolist())) == self.num_capacity_bins, \
            f'Resulting capacity bins size != {self.num_capacity_bins}, bins={bin_edges.tolist()}'

        return bin_edges

    def _get_capacity_bins(self, gate_output: torch.Tensor) -> Union[torch.Tensor, None]:
        """ Generates capacity bins with exponential growing width.

        During training, we encourage tokens to be evenly routed (via aux loss).
        Therefore, generate bins with exponential growing bins width, i.e., bins that are
        closer to the start are smaller and thus have less extra non-required capacity.

        Alignment is required when the bins have to be aligned on a specific value.
        For example:
        1. Configured alignment (capacity_bins_alignment) due to e.g. hardware specific considerations
        2. When the non-experts are using TP and the experts ate not using TP, we
        need to align the bins on TP boundary.

        Args:
            gate_output (torch.Tensor): router gating function output tensor

        Returns:
            bins tensor (torch.Tensor dtype=torch.long)
        """
        if self.capacity_bins is None:
            # calculate optimal and worst case (min and max) tokens per expert
            total_tokens = torch.tensor(self.k * gate_output.shape[0], device=gate_output.device).to(torch.long)
            self.device = gate_output.device
            self.min_tokens_per_expert = torch.ceil(total_tokens / self.num_experts).to(torch.long).item()
            self.max_tokens_per_expert = total_tokens.item()
            # handle bin alignment - maximum between configured alignment and TP (if used)
            tp_alignment = 1
            if groups._get_expert_model_parallel_world_size() == 1 and groups.mpu is not None:
                tp_alignment = groups.mpu.get_tensor_model_parallel_world_size()
            self.alignment = max(self.configured_alignment, tp_alignment)
            # generate bins
            self.capacity_bins = self._generate_bins()
        return self.capacity_bins


def optimize_bins(min_range, bins: torch.Tensor, bins_usage: torch.Tensor, alignment, min_bin_size) -> list:
    """ Optimize MOE capacity bins according to collected bins usage statistics

    The bins are optimized to minimize the cost of binning.
    The cost of each bin is defined as the additional tokens processed in this bin.
    Since we don't have the actual capacities that were mapped to each bin, we use the median of the bin.
    After we calculate the cost of all bins, we iteratively try to replace the lowest and highest cost bins
    with 2 bins: the original highest cost bin and the median of the highest cost bin.
    This way, we keep the number of bins constant while decreasing the overall cost of binning.

    For example:
        Given bins [150, 200, 250, 300] with start of range=100
        And usage  [100, 0,   50,  10 ]

        We first calculate the cost of eac bin:
        Cost:      [25*100, 25*0, 25*50, 25*10] = [2500, 0, 1250, 250]

        Lowest cost bin is 200 (index=1)
        Highest cost bin is 150 (index=0)

        First iteration of optimization:
        Remove bin1 and split bin0 --> [125, 150, 250, 300]
    """

    def align_to(value):
        return int(math.ceil(value / alignment) * alignment)

    # sort bins by their cost of usage (we want to split high cost bins)
    # we assume that for each bin, the cost is 1/2 of its width * usage count
    shifted_bins = torch.cat([torch.tensor([min_range], dtype=bins.dtype, device=bins.device), bins[:-1]])
    width = bins - shifted_bins
    cost = bins_usage * width / 2.0
    sorted_cost = torch.argsort(cost, descending=False, stable=True).tolist()

    # sorted cost is in ascending order
    # min_sort_idx is current index into sorted_cost for candidate bin to be removed
    # max_sort_idx is current index into sorted_cost for candidate bin to be split
    bins = bins.tolist()
    n_bins = len(bins)
    min_sort_idx = 0
    max_sort_idx = n_bins - 1
    new_bins = []
    while min_sort_idx <= max_sort_idx:
        # if same cost, keep all remaining bins and exit
        # this also handles the case of min_sort_idx == max_sort_idx
        min_cost = cost[sorted_cost[min_sort_idx]]
        max_cost = cost[sorted_cost[max_sort_idx]]
        if min_cost == max_cost:
            bin_indexes = sorted_cost[min_sort_idx:max_sort_idx + 1]
            new_bins.extend([bins[idx] for idx in bin_indexes])
            break

        # last bin can't be removed
        min_bin_idx = sorted_cost[min_sort_idx]
        if min_bin_idx == (n_bins - 1):
            new_bins.append(bins[min_bin_idx])
            min_sort_idx += 1
            continue

        # calculate the left & right bin's width of the candidate bin after we split it to 2
        # verify that both left & right will meet the min bin size requirement
        max_bin_idx = sorted_cost[max_sort_idx]
        max_bin_start = min_range if max_bin_idx == 0 else bins[max_bin_idx - 1]
        max_bin_end = bins[max_bin_idx]
        mid_point = (max_bin_start + max_bin_end) // 2
        mid_point = align_to(mid_point)
        left_bin_width = mid_point - max_bin_start
        right_bin_width = max_bin_end - mid_point
        if left_bin_width < min_bin_size or right_bin_width < min_bin_size:
            new_bins.append(bins[max_bin_idx])
            max_sort_idx -= 1
            continue

        # skip min cost bin and split max cost bin
        new_bins.append(mid_point)
        new_bins.append(max_bin_end)
        min_sort_idx += 1
        max_sort_idx -= 1

    # sort the bins in ascending order
    bins = sorted(new_bins)
    return bins
