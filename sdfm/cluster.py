from typing import Iterable
from collections import deque
from dataclasses import dataclass

import ase
import numpy as np
import networkx as nx

from sdfm.template import AtomsTemplate, KEY_TEMPLATE, is_matching_template, BuildingBlock
from sdfm.termination import make_termination


def supertuple(data: Iterable[int]) -> tuple[int]:
    return tuple(sorted(set(data)))


class RecipeBook:
    """Holds a number of templates and corresponding cluster recipes"""

    def __init__(self):
        self.templates, self.recipes = {}, {}

    def get_matching_template(self, atoms: ase.Atoms) -> AtomsTemplate | None:
        """"""
        if KEY_TEMPLATE in atoms.info:
            if (_ := atoms.info[KEY_TEMPLATE]) in self.templates:
                return self.templates[_]

        for label, template in self.templates.items():
            if is_matching_template(template, atoms):
                atoms.info[KEY_TEMPLATE] = label
                return template
        return None

    def add_template(self, template: AtomsTemplate, check: bool = True) -> None:
        """"""
        if check:
            stored_template = self.get_matching_template(template.atoms)
            if stored_template is not None:
                print(f'Template already stored under label "{stored_template.name}"')
                return

        label = template.name
        if label is None:
            label = f"{len(self)}_{template.atoms.get_chemical_formula()}"
        elif label in self.templates:
            label = f"{len(self)}_{label}"
        template.name = template.atoms.info[KEY_TEMPLATE] = label

        self.templates[label] = template
        self.recipes[label] = set()

    def add_recipes(self, template: AtomsTemplate, *recipes: 'ClusterRecipe') -> None:
        """"""
        assert template.name in self.templates, 'Template is not stored..'
        self.recipes[template.name] |= set(recipes)

    def get_recipes(self, template: AtomsTemplate) -> set['ClusterRecipe']:
        assert template.name in self.templates, 'Template is not stored..'
        return self.recipes[template.name]

    def __add__(self, other: 'RecipeBook') -> 'RecipeBook':
        """Merge templates and recipes"""
        assert isinstance(other, RecipeBook), 'Dummy'
        book = RecipeBook()
        for name, template in self.templates.items():
            book.add_recipes(template, *self.recipes[name])
        for name, template in other.templates.items():
            book.add_recipes(template, *other.recipes[name])
        return book

    def __contains__(self, name: str) -> bool:
        return name in self.templates

    def __repr__(self) -> str:
        recipes = tuple([len(r) for r in self.recipes.values()])
        return f'RecipeBook(template: {len(self.templates)}, recipes: {recipes})'

    def __len__(self):
        return len(self.templates)

    @property
    def names(self):
        return set(self.templates)


def assign_template(atoms: ase.Atoms, book: RecipeBook = None, block_size: int = 2) -> AtomsTemplate:
    """"""
    template = None
    if book is not None:
        template = book.get_matching_template(atoms)
    if template is not None:
        return template
    template = AtomsTemplate(atoms)
    template.merge_blocks(block_size)
    if book is not None:
        book.add_template(template, check=False)
    return template


@dataclass
class ClusterRecipe:
    """Container with instructions to extract a cluster from an AtomsTemplate"""
    # TODO: more instructions (keep cell / transform coordinates / how to terminate...)
    ids: tuple[int]
    ids_core: tuple[int]
    block_ids: tuple[int]
    block_ids_core: tuple[int]
    broken_bonds: np.ndarray[int]

    @classmethod
    def from_blocks(cls, template: AtomsTemplate, block_ids: list[int], core_block_ids: list[int]) -> 'ClusterRecipe':
        """Construct recipe from template and block indices"""
        b_ids, b_ids_core = set(block_ids), set(core_block_ids)
        assert all(_ in b_ids for _ in b_ids_core)
        ids = set().union(*[template.block_ids[idx] for idx in b_ids])
        ids_core = set().union(*[template.block_ids[idx] for idx in b_ids_core])
        broken_bonds = np.array(list(nx.edge_boundary(template.graph, ids)))
        kwargs = dict(ids=ids, ids_core=ids_core, block_ids=b_ids, block_ids_core=b_ids_core)
        return cls(**{k: supertuple(v) for k, v in kwargs.items()}, broken_bonds=broken_bonds)

    # @classmethod
    # def from_indices(cls, template: AtomsTemplate, ids: list[int], core_ids: list[int]):
    #     # TODO: what is the use case for this method?
    #     ids_to_block = {idx: block_idx for block_idx, ids in template.block_ids for idx in ids}
    #     block_ids = sorted(set([ids_to_block[idx] for idx in ids]))
    #     core_block_idx = set([ids_to_block[idx] for idx in core_ids])[0]
    #     return cls.from_blocks(template, block_ids, core_block_idx)

    def __len__(self): return len(self.ids)

    def __eq__(self, other) -> bool:
        return self.__hash__() == other.__hash__()

    def __hash__(self) -> int:
        return hash((self.ids, self.block_ids, self.ids_core, self.block_ids_core))


def extract_cluster(template: AtomsTemplate, recipe: ClusterRecipe, atoms: ase.Atoms):
    """Extract a cluster from atoms using template and recipe"""
    # TODO: check that atoms corresponds with template?
    # TODO: log template name?
    if len(recipe) == len(atoms):  # not a cluster
        return atoms

    # insert sample atoms
    pos_atoms_copy = atoms.get_positions()
    template.atoms, template_atoms = atoms, template.atoms

    # make a single fragment
    blocks = [template.blocks[id_b] for id_b in recipe.block_ids]
    blocks_core = [template.blocks[id_b] for id_b in recipe.block_ids_core]
    unite_blocks(template, blocks, blocks_core[0])  # TODO: this will not work for disconnected clusters

    # create cluster
    cluster = template.atoms[recipe.ids]
    cluster.arrays['ref_idx'] = np.array(recipe.ids)
    cluster.arrays['core'] = np.isin(recipe.ids, recipe.ids_core, assume_unique=True)

    # create termination
    pos = template.pos[recipe.broken_bonds]
    termination = make_termination(atoms, pos_in=pos[:, 0], pos_out=pos[:, 1])
    termination.arrays |= {'ref_idx': - np.ones(len(termination))}
    cluster += termination
    cluster.center(vacuum=0)  # move to origin
    cluster.cell, cluster.pbc = None, False

    # restore original template and atoms
    template.atoms, atoms.positions = template_atoms, pos_atoms_copy

    return cluster


def unite_blocks(template: AtomsTemplate, blocks: list[BuildingBlock], initial: BuildingBlock) -> None:
    """Shift connected blocks across periodic boundary to create a unified fragment"""
    subgraph, cell = template.graph_cg.subgraph(blocks), template.cell

    # unite atoms per block
    for block in blocks:
        block.mic()

    # shift blocks to appropriate periodic copy
    check = deque([initial])
    moved = set(check)
    while len(moved) < len(blocks):
        block = check.popleft()
        center = cell.scaled_positions(block.pos)
        neighbours = nx.node_boundary(subgraph, {block})
        for b in {b for b in neighbours if b not in moved}:
            pos = ase.geometry.wrap_positions(b.pos.reshape(1, -1), cell, center=center)
            b.move_to(pos)
            moved.add(b)
            check.append(b)
    return
