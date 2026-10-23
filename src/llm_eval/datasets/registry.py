"""In-memory registry of versioned datasets."""

from __future__ import annotations

from llm_eval.models import Dataset


class DatasetRegistry:
    """Store datasets by name and version.

    Datasets are addressed as ``name`` (resolving to the latest registered
    version) or as explicit ``name@version`` keys.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._by_key: dict[str, Dataset] = {}
        self._latest: dict[str, str] = {}

    @staticmethod
    def _key(name: str, version: str) -> str:
        """Compose a storage key.

        Args:
            name: Dataset name.
            version: Dataset version.

        Returns:
            The canonical ``name@version`` key.
        """

        return f"{name}@{version}"

    def register(self, dataset: Dataset) -> Dataset:
        """Register a dataset version.

        Args:
            dataset: Dataset to store.

        Returns:
            The registered dataset.
        """

        self._by_key[self._key(dataset.name, dataset.version)] = dataset
        self._latest[dataset.name] = dataset.version
        return dataset

    def get(self, name: str, version: str | None = None) -> Dataset:
        """Retrieve a dataset by name and optional version.

        Args:
            name: Dataset name.
            version: Explicit version, or ``None`` for the latest.

        Returns:
            The matching dataset.

        Raises:
            KeyError: If no dataset matches.
        """

        resolved = version or self._latest.get(name)
        if resolved is None:
            raise KeyError(f"unknown dataset: {name}")
        key = self._key(name, resolved)
        if key not in self._by_key:
            raise KeyError(f"unknown dataset version: {key}")
        return self._by_key[key]

    def resolve(self, reference: str) -> Dataset:
        """Resolve a ``name`` or ``name@version`` reference.

        Args:
            reference: Dataset reference string.

        Returns:
            The matching dataset.

        Raises:
            KeyError: If no dataset matches.
        """

        if "@" in reference:
            name, version = reference.split("@", 1)
            return self.get(name, version)
        return self.get(reference)

    def list_names(self) -> list[str]:
        """List registered dataset names.

        Returns:
            Sorted unique dataset names.
        """

        return sorted(self._latest)

    def list_versions(self, name: str) -> list[str]:
        """List registered versions for a dataset.

        Args:
            name: Dataset name.

        Returns:
            Sorted version strings, empty when the name is unknown.
        """

        prefix = f"{name}@"
        return sorted(key[len(prefix):] for key in self._by_key if key.startswith(prefix))

    def __len__(self) -> int:
        """Return the number of registered dataset versions."""

        return len(self._by_key)

    def __contains__(self, reference: object) -> bool:
        """Return whether a reference resolves to a dataset."""

        if not isinstance(reference, str):
            return False
        try:
            self.resolve(reference)
        except KeyError:
            return False
        return True

    def clear(self) -> None:
        """Remove every registered dataset."""

        self._by_key.clear()
        self._latest.clear()


_REGISTRY = DatasetRegistry()


def get_registry() -> DatasetRegistry:
    """Return the process-wide dataset registry.

    Returns:
        The shared :class:`DatasetRegistry` singleton.
    """

    return _REGISTRY
