from collections.abc import Mapping
from typing import Any, TypeVar, overload
from numbers import Integral


import numpy as np
import pandas as pd

from xarray import Index
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import IndexSelResult
from xarray.core.variable import Variable

__all__ = [
    "DimensionInterval",
]

_T = TypeVar("_T")


def merge_sel_results(results: list[IndexSelResult]) -> IndexSelResult:
    # all_dims_count = Counter([dim for res in results for dim in res.dim_indexers])
    # duplicate_dims = {k: v for k, v in all_dims_count.items() if v > 1}

    # if duplicate_dims:
    #     # TODO: this message is not right when combining indexe(s) queries with
    #     # location-based indexing on a dimension with no dimension-coordinate (failback)
    #     fmt_dims = [
    #         f"{dim!r}: {count} indexes involved"
    #         for dim, count in duplicate_dims.items()
    #     ]
    #     raise ValueError(
    #         "Xarray does not support label-based selection with more than one index "
    #         "over the following dimension(s):\n"
    #         + "\n".join(fmt_dims)
    #         + "\nSuggestion: use a multi-index for each of those dimension(s)."
    #     )

    dim_indexers = {}
    indexes = {}
    variables = {}
    drop_coords = []
    drop_indexes = []
    rename_dims = {}

    for res in results:
        dim_indexers.update(res.dim_indexers)
        indexes.update(res.indexes)
        variables.update(res.variables)
        drop_coords += res.drop_coords
        drop_indexes += res.drop_indexes
        rename_dims.update(res.rename_dims)

    return IndexSelResult(
        dim_indexers, indexes, variables, drop_coords, drop_indexes, rename_dims
    )


class DimensionInterval(Index):
    _continuous_index: PandasIndex
    _interval_index: PandasIndex

    def __init__(
        self,
        continuous_index: PandasIndex,
        interval_index: PandasIndex,
        continuous_dim_name: str,
        interval_dim_name: str,
    ):
        assert isinstance(interval_index.index, pd.IntervalIndex)
        assert isinstance(continuous_index.index, pd.Index)
        self._continuous_index = continuous_index
        self._interval_index = interval_index

        self._continuous_name = continuous_dim_name
        self._interval_name = interval_dim_name

    @classmethod
    def from_variables(cls, variables, *, options):
        assert len(variables) == 2

        # TODO:
        # extract interval_name as the dim for interval
        # e.g. word or intervals
        indexes = {
            k: PandasIndex.from_variables({k: v}, options=options)
            for k, v in variables.items()
        }

        # TODO: are we enforcing contiguousness here or allowing disjoint intervals?
        c_dim = "time"
        i_dim = "intervals"
        return cls(
            # more hardocoding - TODO: improve
            continuous_index=indexes[c_dim],
            interval_index=indexes[i_dim],
            continuous_dim_name=c_dim,
            interval_dim_name=i_dim,
        )

    def create_variables(self, variables):
        idx_variables = {}

        for index in (self._continuous_index, self._interval_index):
            idx_variables.update(index.create_variables(variables))

        return idx_variables

    @overload
    @staticmethod
    def _no_zero_slice(indexer: int | np.integer) -> slice: ...
    @overload
    @staticmethod
    def _no_zero_slice(indexer: _T) -> _T: ...
    @staticmethod
    def _no_zero_slice(indexer: int | np.integer | _T) -> slice | _T:
        if isinstance(indexer, (int, np.integer)):
            return slice(indexer, indexer + 1)
        return indexer

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> "DimensionInterval | None":
        print(indexers)
        # Get indexers for each dimension this index manages
        continuous_indexer = indexers.get(self._continuous_name)
        interval_indexer = indexers.get(self._interval_name)

        new_cont_index = self._continuous_index
        new_interval_index = self._interval_index

        def co_slice(
            leader_index: PandasIndex,
            follower_index: PandasIndex,
            leader_name: str,
            follower_name: str,
            indexer: slice | Integral,
        ):
            if isinstance(indexer, Integral):
                # Need to take special care that we don't eliminate a dim here
                # because we can't return from isel that only one of our two is no longer here
                # we basically have to return a version of self
                new_leader_index = leader_index.isel(
                    {leader_name: slice(indexer, indexer + 1)}
                )
            else:
                new_leader_index = leader_index.isel({leader_name: indexer})
            if isinstance(leader_index.index, pd.IntervalIndex):
                interval = leader_index.index[indexer]
                follow_slice = slice(interval.left, interval.right)
            else:
                follow_slice = slice(
                    new_leader_index.index.values.min(),
                    new_leader_index.index.values.max(),
                )

            follow_extremes = follower_index.sel(
                labels={follower_name: follow_slice}
            ).dim_indexers[follower_name]
            new_follower_index = follower_index.isel({follower_name: follow_extremes})
            return new_leader_index, new_follower_index

        if continuous_indexer is not None and interval_indexer is not None:
            # what to do here?
            # what if they are in conflict i.e. only partially overlapping?
            # i guess we take the most restrictive possible approach
            # so do interval then do contintuous
            # can return a chained self.isel?

            new_interval_index = self.isel(
                {self._interval_name: interval_indexer}
            )._interval_index
            new_cont_index = self.isel(
                {self._continuous_name: continuous_indexer}
            )._continuous_index

            # there are definitely some cases that above will not handle well. in particular double isel selections
            # will almost certainly break
            # can't just chain them. becuase then you end up double slicing time in potentially incompatible ways
            # so time ends up too short
            # return self.isel({self._interval_name: interval_indexer}).isel(
            #     {self._continuous_name: continuous_indexer}
            # )
        elif continuous_indexer is not None:
            # print("CONT")
            # if new_cont_index is None:
            #     new_cont_index = self._continuous_index.isel
            #     return None
            continuous_indexer = self._no_zero_slice(continuous_indexer)
            if isinstance(continuous_indexer, (slice, Integral)):
                new_cont_index, new_interval_index = co_slice(
                    self._continuous_index,
                    self._interval_index,
                    self._continuous_name,
                    self._interval_name,
                    continuous_indexer,
                )
            else:
                # TODO: an array (or maybe other things?)
                raise NotImplementedError

        elif interval_indexer is not None:
            # print("INT")
            # interval_indexer = self._no_zero_slice(interval_indexer)
            if isinstance(interval_indexer, (slice, Integral)):
                new_interval_index, new_cont_index = co_slice(
                    self._interval_index,
                    self._continuous_index,
                    self._interval_name,
                    self._continuous_name,
                    interval_indexer,
                )
            else:
                raise NotImplementedError

        # kinda wish i could also return a tuple here making one of these NOne
        # but keeping the other one
        # if new_interval_index is not None and new_cont_index is not None:
        # This is basically GH issue:
        # https://github.com/pydata/xarray/issues/10477
        assert new_cont_index is not None
        assert new_interval_index is not None
        return DimensionInterval(
            continuous_index=new_cont_index,
            interval_index=new_interval_index,
            continuous_dim_name=self._continuous_name,
            interval_dim_name=self._interval_name,
        )

    def should_add_coord_to_array(self, name, var, dims) -> bool:
        # TODO: this cannot be right....
        # passes name of dataarray
        # this gets passed a single coord
        # then decide what to do with it
        # dims is dimension of dataarray, not the coord (which is var here)
        return True

    def sel(self, labels, **kwargs):
        results = []

        # cannot only handle this complexity in isel, because if you select on the continuous dimension
        # to a point we can't properly drop the interval coord without passing both through the indexing here
        # e.g. ds.sel(time=10) . we have to pass both indexers from here. we also need to indepdently handle that case
        # inside of isel. Can probably consolidate the code for handling this in the future.
        for k in labels.keys():
            if k == self._continuous_name:
                cont_res = self._continuous_index.sel({k: labels[k]}, **kwargs)
                results.append(cont_res)
                values = cont_res.dim_indexers[self._continuous_name]
                if isinstance(values, Integral):
                    intervals = self._interval_index.sel(
                        labels={
                            self._interval_name: self._continuous_index.index[values]
                        }
                    )
                    results.append(intervals)
                    self._continuous_index.index
            elif k == self._interval_name:
                int_res = self._interval_index.sel({k: labels[k]}, **kwargs)
                results.append(int_res)
                which_intervals = int_res.dim_indexers[self._interval_name]
                if isinstance(which_intervals, Integral):
                    interval: Any = self._interval_index.index[which_intervals]
                    # TODO: handle closed left, right both etc
                    cont_slice = self._continuous_index.sel(
                        {self._continuous_name: slice(interval.left, interval.right)}
                    )
                    results.append(cont_slice)
                else:
                    raise NotImplementedError

        res = merge_sel_results(results)

        return res
