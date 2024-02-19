# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Union
from deepspeed import comm as dist
from deepspeed.utils import groups


class CapacityBins:
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

    def __init__(self, k: int, num_experts: int, num_capacity_bins: int, capacity_bins_exp_base: float,
                 capacity_bins_alignment: int) -> None:
        self.k = k
        self.num_experts = num_experts
        self.num_capacity_bins = num_capacity_bins
        self.capacity_bins_exp_base = capacity_bins_exp_base
        self.configured_alignment = capacity_bins_alignment
        self.device = None
        self.min_tokens_per_expert = None
        self.max_tokens_per_expert = None
        self.alignment = None
        self.min_bin_size = None
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
