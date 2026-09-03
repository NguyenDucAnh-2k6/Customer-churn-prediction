"""
Dataset Registry for discovering and instantiating datasets.
"""

from typing import Any, Callable, Dict, List, Type

from src.data.base import BaseDataset
from src.data.timeseries import TimeSeriesDataset, GroupStratifiedTimeSeriesDataset
from src.data.static import StaticDataset
from src.data.latest import PreSplitLatestDataset, GroupStratifiedLatestDataset
from src.data.pit import PointInTimeTimeSeriesDataset
from src.data.round3 import Round3Dataset
from src.data.round3_timeseries import Round3TimeSeriesDataset


class DatasetRegistry:
    """Registry pattern for discovering and instantiating datasets."""
    _registry: Dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseDataset]], Type[BaseDataset]]:
        """Decorator to register a new dataset class."""
        def decorator(subclass: Type[BaseDataset]) -> Type[BaseDataset]:
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseDataset:
        """Instantiate a registered dataset by name."""
        key = name.lower()
        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Dataset '{name}' is not registered. Available datasets: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of all registered dataset names."""
        return sorted(list(cls._registry.keys()))


# Register all built-in datasets
DatasetRegistry.register("timeseries")(TimeSeriesDataset)
DatasetRegistry.register("timeseries_group")(GroupStratifiedTimeSeriesDataset)
DatasetRegistry.register("group_stratified")(GroupStratifiedTimeSeriesDataset)
DatasetRegistry.register("static")(StaticDataset)
DatasetRegistry.register("latest")(PreSplitLatestDataset)
DatasetRegistry.register("latest_group")(GroupStratifiedLatestDataset)
DatasetRegistry.register("presplit")(PreSplitLatestDataset)
DatasetRegistry.register("churn_team")(PreSplitLatestDataset)
DatasetRegistry.register("pit")(PointInTimeTimeSeriesDataset)
DatasetRegistry.register("point_in_time")(PointInTimeTimeSeriesDataset)
DatasetRegistry.register("timeseries_pit")(PointInTimeTimeSeriesDataset)
DatasetRegistry.register("round3")(Round3Dataset)
DatasetRegistry.register("round_3")(Round3Dataset)
DatasetRegistry.register("r3")(Round3Dataset)
DatasetRegistry.register("round3_timeseries")(Round3TimeSeriesDataset)
DatasetRegistry.register("r3_timeseries")(Round3TimeSeriesDataset)
DatasetRegistry.register("timeseries_round3")(Round3TimeSeriesDataset)
