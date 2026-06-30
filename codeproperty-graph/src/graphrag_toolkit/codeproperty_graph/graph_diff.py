# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph Diff — compare two CPG states and compute delta."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DiffResult:
    """Result of comparing two method signature sets."""

    added: Dict[str, str] = field(default_factory=dict)      # full_name → hash (new methods)
    removed: Dict[str, str] = field(default_factory=dict)    # full_name → hash (deleted methods)
    modified: Dict[str, str] = field(default_factory=dict)   # full_name → new_hash (body changed)
    unchanged: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified)

    @property
    def summary(self) -> str:
        return f"+{len(self.added)} -{len(self.removed)} ~{len(self.modified)} ={self.unchanged}"


class GraphDiff:
    """Compare method signatures between two CPG exports."""

    @staticmethod
    def compare(
        previous: Dict[str, str],
        current: Dict[str, str],
    ) -> DiffResult:
        """Compare previous vs current method_signatures dicts.

        Args:
            previous: {full_name: hash} from manifest
            current: {full_name: hash} from new export

        Returns:
            DiffResult with added/removed/modified/unchanged counts
        """
        prev_keys = set(previous.keys())
        curr_keys = set(current.keys())

        added = {k: current[k] for k in curr_keys - prev_keys}
        removed = {k: previous[k] for k in prev_keys - curr_keys}

        modified = {}
        unchanged = 0
        for k in prev_keys & curr_keys:
            if previous[k] != current[k]:
                modified[k] = current[k]
            else:
                unchanged += 1

        return DiffResult(
            added=added,
            removed=removed,
            modified=modified,
            unchanged=unchanged,
        )
