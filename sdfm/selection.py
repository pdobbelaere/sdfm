"""
Select atoms or building blocks through various rules
"""
import copy
from typing import Iterable

import ase.geometry
import ase.symbols
import numpy as np
import networkx as nx

from sdfm.manip.template import AtomsTemplate


class AtomSelection:
    """"""

    def __init__(self, template: AtomsTemplate, level: str = 'blocks'):
        self._template = template
        self.blocks = np.zeros(len(template.blocks), dtype=bool)
        self.atoms = np.zeros(len(template.atoms), dtype=bool)
        self.map_block_to_atoms = {i: ids for i, ids in enumerate(template.block_ids)}
        self.map_atom_to_block = {idx: k for k, v in self.map_block_to_atoms.items() for idx in v}
        self.level = level

    def select(self, *rules: 'SelectionRule', add: bool = True):
        """Add to or remove from selection based on rules. When multiple rules are provided, use their intersection."""
        assert self.is_consistent, 'Selection is no longer consistent with template. Reinitialise..'
        assert all(isinstance(rule, SelectionRule) for rule in rules)

        active_mask, active_ids = self.active_mask, self.active_ids
        candidates = set(np.flatnonzero(~active_mask) if add else active_ids)
        for rule in rules:
            candidates = rule.select(candidates, self._template, self.level, active_ids)
        for idx in candidates:
            active_mask[idx] = add

    def modify(self, rule: 'ModificationRule'):
        """Modify the current selection based on rule."""
        ids = rule.modify(self._template, self.level, self.active_ids)
        self.active_mask[:] = False
        self.active_mask[ids] = True

    def reset(self) -> None:
        self.blocks[:] = self.atoms[:] = False

    def copy(self) -> 'AtomSelection':
        """"""
        new = copy.copy(self)
        new.blocks, new.atoms = new.blocks.copy(), new.atoms.copy()  # uncouple selection arrays
        return new

    def get_atoms_mask(self) -> np.ndarray[bool]:
        """"""
        if self.level == 'atoms':
            return self.atoms.copy()
        # atoms follow blocks
        arr = np.zeros_like(self.atoms, dtype=bool)
        for id_b in self.active_ids:
            arr[self.map_block_to_atoms[id_b]] = True
        return arr

    @property
    def active_mask(self) -> np.ndarray[bool]:
        return self.atoms if self.level == 'atoms' else self.blocks

    @property
    def active_ids(self) -> np.ndarray[int]:
        return np.flatnonzero(self.active_mask)

    @property
    def n_blocks(self) -> int:
        return self.blocks.size

    @property
    def n_atoms(self) -> int:
        return self.atoms.size

    @property
    def n_selected(self):
        return self.active_mask.sum()

    @property
    def is_consistent(self) -> bool:
        return self.n_atoms == len(self._template) and self.n_blocks == len(self._template.blocks)

    def toggle_level(self):
        """"""
        if self.level == 'atoms':
            # select blocks if any atom was in selection
            for id_b in set(self.map_atom_to_block[i] for i in self.active_ids):
                self.blocks[id_b] = True
            self.atoms[:] = False
        elif self.level == 'blocks':
            self.atoms[:] = self.get_atoms_mask()
            self.blocks[:] = False

        self.level = 'atoms' if self.level == 'blocks' else 'blocks'

    def __repr__(self):
        n_a, n_b, lvl = self.n_atoms, self.n_blocks, self.level
        return f'Selection(level={lvl}, n_atoms={n_a}, n_blocks={n_b}, selected={self.n_selected})'

    def __str__(self):
        return self.__repr__()


class SelectionRule:
    """Baseclass for rules to select atoms or blocks"""

    def __init__(self, invert: bool = False):
        self.invert = invert

    def select(self, candidates: set[int], template: AtomsTemplate, level: str, active_ids: Iterable[int]) -> set[int]:
        """Return a set with block or atom indices that fulfill this rule"""
        if level == 'blocks':
            ids = self.select_blocks(candidates, template, active_ids)
        else:
            ids = self.select_atoms(candidates, template, active_ids)
        return candidates.intersection(ids) if not self.invert else candidates.difference(ids)

    def select_blocks(self, candidates: set[int], template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        raise NotImplementedError()

    def select_atoms(self, candidates: set[int], template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        raise NotImplementedError()


class SelectByIndex(SelectionRule):
    """Select based on index"""

    def __init__(self, *ids: int, **kwargs):
        super().__init__(**kwargs)
        self.ids = ids

    def select_blocks(self, *args) -> Iterable[int]:
        return self.ids

    def select_atoms(self, *args) -> Iterable[int]:
        return self.ids


class SelectByElement(SelectionRule):
    """Select based on a specified element"""

    def __init__(self, element: int | str, **kwargs):
        super().__init__(**kwargs)
        self.number = ase.symbols.symbols2numbers([element])[0]

    def select_blocks(self, candidates: set[int], template: AtomsTemplate, *args) -> Iterable[int]:
        return [idx for idx in candidates if self.number in np.unique(template.blocks[idx].numbers)]

    def select_atoms(self, candidates: set[int], template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        # TODO: check computational cost of this
        return [idx for idx in candidates if self.number == template.atoms[idx].number]


class SelectBySize(SelectionRule):
    """Select blocks based on size"""

    def __init__(self, size: int, **kwargs):
        super().__init__(**kwargs)
        self.n = size

    def select_blocks(self, candidates: set[int], template: AtomsTemplate, *args) -> Iterable[int]:
        return [idx for idx in candidates if len(template.blocks[idx]) == self.n]


class SelectByConnectivity(SelectionRule):
    """Select based on the number of graph neighbours"""

    def __init__(self, neighbours: int, **kwargs):
        super().__init__(**kwargs)
        self.n = neighbours

    def select_blocks(self, candidates: set[int], template: AtomsTemplate, *args) -> Iterable[int]:
        return [idx for idx in candidates if len(template.blocks[idx].broken_bonds) == self.n]

    def select_atoms(self, candidates: set[int], template: AtomsTemplate, *args) -> Iterable[int]:
        labels = [idx for idx in candidates if len(template.graph[idx]) == self.n]
        return template.labels_to_ids(labels)


class SelectByChance(SelectionRule):
    """Select based on random chance"""

    def __init__(self, fraction: float = None, amount: int = None, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        assert (fraction is None) != (amount is None)  # xor
        self.rng, self.p, self.n = np.random.default_rng(seed), fraction, amount

    def select_blocks(self, candidates: set[int], *args) -> Iterable[int]:
        n = self.n or round(self.p * len(candidates))
        return self.rng.choice(tuple(candidates), size=n, replace=False)

    def select_atoms(self, candidates: set[int], *args) -> Iterable[int]:
        return self.select_blocks(candidates)


class SelectByAdjacency(SelectionRule):
    """Select based on graph adjacency"""

    def __init__(self, order: int, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def select_blocks(self, candidates: set[int], template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        blocks = {template.blocks[idx] for idx in active_ids}
        for _ in range(self.order):
            blocks |= nx.node_boundary(template.graph_cg, blocks)
        return candidates.intersection([template.blocks.index(b) for b in blocks])

    def select_atoms(self, candidates: set[int], template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        labels = set(template.atoms.arrays['labels'][np.array(tuple(active_ids))])
        for _ in range(self.order):
            labels |= nx.node_boundary(template.graph, labels)
        return candidates.intersection(template.labels_to_ids(list(labels)))


class SelectInSphere(SelectionRule):
    """TODO: this feels not quite right"""

    def __init__(self, radius: float, center: tuple[float, float, float], **kwargs):
        super().__init__(**kwargs)
        self.radius, self.center = radius, center
        assert len(center) == 3

    def select_blocks(self, candidates: set[int], template: AtomsTemplate, *args) -> Iterable[int]:
        pos = np.array([b.pos for b in template.blocks])
        return self._select(candidates, pos, template.cell, all(template.pbc))

    def select_atoms(self, candidates: set[int], template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        pos = template.atoms.positions
        return self._select(candidates, pos, template.cell, all(template.pbc))

    def _select(self, candidates: set[int], pos: np.ndarray, cell: ase.atoms.Cell, pbc: bool) -> Iterable[int]:
        candidates = list(candidates)
        _, dist = ase.geometry.get_distances(pos[candidates], self.center, cell, pbc)
        ids = np.flatnonzero(dist.flatten() <= self.radius)
        return [candidates[idx] for idx in ids]


class SelectByCycle(SelectionRule):
    """Select a connected cycle of blocks with specified length."""

    # TODO: this feels not quite right
    def __init__(self, length: int, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        assert length > 0

    def select_blocks(self, candidates: set[int], template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        cycles = nx.simple_cycles(template.graph_cg, length_bound=self.length)
        for cycle in cycles:
            if len(cycle) == self.length:
                return [template.blocks.index(b) for b in cycle]


class ModificationRule:
    """Baseclass for rules to modify AtomsSelection"""

    # TODO: Mixin?
    def modify(self, template: AtomsTemplate, level: str, active_ids: Iterable[int]) -> list[int]:
        if level == 'blocks':
            ids = self.modify_blocks(template, active_ids)
        else:
            ids = self.modify_atoms(template, active_ids)
        return list(ids)

    def modify_blocks(self, template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        raise NotImplementedError()

    def modify_atoms(self, template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        raise NotImplementedError()


class ReduceToComponent(ModificationRule):
    """TODO: extend this"""

    def __init__(self, idx: int = None):
        self.idx = idx

    def modify_blocks(self, template: AtomsTemplate, active_ids: Iterable[int]) -> Iterable[int]:
        blocks = [template.blocks[idx] for idx in active_ids]
        subgraph = template.graph_cg.subgraph(blocks)
        components = list(nx.connected_components(subgraph))
        if self.idx is None:
            # find largest component
            sizes = [sum(len(block) for block in comp) for comp in components]
            component = components[sizes.index(max(sizes))]
        else:
            # find component containing block idx
            block = template.blocks[self.idx]
            component = [comp for comp in components if block in comp][0]  # TODO: will fail if idx not active
        return [template.blocks.index(block) for block in component]

#
# def select_within_volume(template: AtomsTemplate, selection: AtomSelection,
#                          vmin: tuple[float, float, float] = None, vmax: tuple[float, float, float] = None,
#                          invert: bool = False):
#     """"""
#     # TODO: periodic boundaries?
#     print(vmin, vmax)
#     vmin = np.array(vmin) if vmin is not None else - np.ones(3) * np.inf
#     vmax = np.array(vmax) if vmax is not None else + np.ones(3) * np.inf
#     print(vmin, vmax)
#
#     block_ids = [i for i, b in enumerate(template.blocks) if (b.pos > vmin).all() * (b.pos < vmax).all()]
#     if invert:
#         block_ids = set(range(len(template.blocks))).difference(block_ids)
#     selection.update(block_ids=block_ids)
#
#
# def select_blocks_in_radius(template: AtomsTemplate, core_block_idx: int, radius: float) -> ClusterRecipe:
#     """Makes a recipe for all blocks in template within a sphere of radius around the chosen core block"""
#     # TODO: seems like more of a selection method? or a rule for ClusterRuleRecipe?
#     # create dummy CG representation of blocks in atoms to compute distances
#     dummy = ase.Atoms(numbers=np.zeros(len(template.blocks)),
#                       positions=[block.pos for block in template.blocks],
#                       cell=template.cell, pbc=True)
#     dist = dummy.get_distances(core_block_idx, np.arange(len(dummy)), mic=True)
#
#     return ClusterRecipe.from_blocks(template, blocks_ids=list(np.flatnonzero(dist < radius)),
#                                      core_block_idx=core_block_idx)
#
