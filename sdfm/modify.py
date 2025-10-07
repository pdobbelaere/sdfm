import pickle
from dataclasses import dataclass, field
from typing import Iterable

import ase
import numpy as np
import networkx as nx

from sdfm.template import AtomsTemplate, BuildingBlock
from sdfm.termination import make_termination
from sdfm.utils import compute_inertia, match_point_clouds


# class State(Enum):
#     CONNECTED = 1
#     DANGLING = 0
#     LOOSE = -1


# class CGBond:
#     """Bond between atoms that connect two building blocks"""
#     labels: tuple[int, int]
#     ids: tuple[int, int]
#     pos: np.ndarray
#
#     def get_state(self, template: AtomsTemplate) -> State:
#         in1, in2 = self.labels[0] in template.graph, self.labels[1] in template.graph
#         if in1 and in2:
#             return State.CONNECTED
#         return State(in1 + in2)
#
#     @staticmethod
#     def create_from_template(template: ExtendedAtomsTemplate) -> set['CGBond']:
#         pass


@dataclass(frozen=True)
class DanglingBond:
    """Broken bonds within a block or template. These should point outwards"""
    labels: tuple[int, int]
    ids: tuple[int, int]
    pos: np.ndarray = field(hash=False, compare=False)

    def is_dangling(self, template: AtomsTemplate) -> bool:
        """Only one of the atom labels can still be in the graph"""
        return (self.labels[0] in template.graph) != (self.labels[1] in template.graph)  # xor

    def invert(self) -> 'DanglingBond':
        """"""
        return DanglingBond(self.labels[::-1], self.ids[::-1], self.pos[::-1])

    @property
    def vec(self) -> np.ndarray:
        return (vec := self.pos[1] - self.pos[0]) / np.linalg.norm(vec)

    @classmethod
    def from_template(cls, template: AtomsTemplate, label_in: int, label_out: int):
        """"""
        idx_in, idx_out = template.labels_to_ids([label_in, label_out])
        pos_in = template.pos[idx_in]
        pos_out = pos_in + template.atoms.get_distance(idx_in, idx_out, vector=True, mic=True)
        return cls((label_in, label_out), (idx_in, idx_out), np.array([pos_in, pos_out]))

    @staticmethod
    def from_block(block: BuildingBlock, invert: bool = False) -> list['DanglingBond']:
        """"""
        # block.mic()                     # get all dangling bonds in a single periodic image
        bonds = [DanglingBond.from_template(block.template, *l) for l in block.broken_bonds]
        return [bond.invert() for bond in bonds] if invert else bonds


class ExtendedAtomsTemplate(AtomsTemplate):
    """Additional functionality for structure manipulation"""
    ghost_atoms: ase.Atoms
    termination: ase.Atoms
    dangling_bonds: set[DanglingBond]
    _label_to_idx_map: dict[int, int]  # converts labels back to indices

    def __init__(self, atoms: ase.Atoms, name: str = None, graph: nx.Graph = None):
        self.ghost_atoms = self.termination = self._label_to_idx_map = None
        super().__init__(atoms, name, graph)
        self.dangling_bonds = set()
        # tmp = CGBond.create_from_template(self)

    def get_atoms(self, ghost_atoms: bool = False, termination: bool = False, wrap: bool = False) -> ase.Atoms:
        """Return parts of this template, for visualisation purposes and such"""
        atoms = self.atoms.copy()
        if ghost_atoms:
            self._make_ghost_atoms()
            atoms += self.ghost_atoms
        if termination:
            self._make_termination()
            atoms += self.termination
        if wrap:
            atoms.wrap(pbc=True)
        return atoms

    def labels_to_ids(self, labels: list[int]) -> list[int]:
        if self._label_to_idx_map is None:
            self._label_to_idx_map = {l: idx for idx, l in enumerate(self.labels)}
        return [self._label_to_idx_map[l] for l in labels]

    def copy(self) -> 'ExtendedAtomsTemplate':
        return pickle.loads(pickle.dumps(self))  # much faster than deepcopy

    def remove_blocks(self, block_ids: Iterable[int]) -> None:
        """"""
        blocks = [self.blocks[i] for i in block_ids]
        labels = sum([block.labels for block in blocks], [])
        ids = self.labels_to_ids(labels)
        dangling_bonds = sum([DanglingBond.from_block(b, invert=True) for b in blocks], [])

        # remove blocks and atoms from template and graphs
        del self.atoms[ids]
        self.graph.remove_nodes_from(labels)
        self.graph_cg.remove_nodes_from(blocks)
        self.blocks = sorted(self.graph_cg.nodes, key=len, reverse=True)
        assert len(self.bonds_cg) == len(self.connections), 'CG mapping bugged out'

        # ensure building blocks and dangling bonds remain consistent with remaining template
        self._label_to_idx_map = {l: i for i, l in enumerate(self.labels)}
        for block in self.blocks:
            block.reset()
        if self.dangling_bonds is not None:
            dangling_bonds = self.dangling_bonds.union(dangling_bonds)
        self.dangling_bonds = set(bond for bond in dangling_bonds if bond.is_dangling(self))

        return

    def insert_block(self, new_block: 'StandaloneBuildingBlock', dangling_bonds: list[DanglingBond]) -> None:
        """"""
        # TODO: this is pretty slow..
        atoms = new_block.align_atoms(dangling_bonds)

        # create new atom labels and relabel block
        # graph label 'order' is different from atoms label order
        labels = self._make_node_labels(size=len(atoms), start=self.labels.max() + 1)
        label_map = {l_old: l_new for l_old, l_new in zip(new_block.graph.nodes, labels)}
        graph = nx.relabel_nodes(new_block.graph, label_map)
        atoms.set_array('labels', [label_map[l] for l in atoms.arrays['labels']])
        self._label_to_idx_map |= {label: idx + len(self) for idx, label in enumerate(atoms.arrays['labels'])}

        # combine template graph with new block graph
        assert all(b in self.dangling_bonds for b in dangling_bonds)
        bonds = [(old.labels[0], label_map[new.labels[0]])
                 for old, new in zip(dangling_bonds, new_block.dangling_bonds)]
        graph.add_edges_from(bonds, breakable=True)
        self.graph = nx.compose(self.graph, graph)

        # make copy to avoid modifying while iterating when dangling_bonds is self.dangling_bonds
        for dangling_bond in set(dangling_bonds):
            self.dangling_bonds.remove(dangling_bond)

        self.atoms += atoms
        self.blocks.append(block := BuildingBlock(self, labels))

        # add bonds between cg graph and new block  TODO: clean this up?
        cg_bonds = [(block, next((b for b in self.blocks if bond[0] in b.graph))) for bond in bonds]
        self.graph_cg.add_edges_from(cg_bonds)
        for b in [_[1] for _ in cg_bonds]:
            b.reset()

        self._check_sanity()

        return

    def swap_blocks(self, block_ids: Iterable[int], new_block: 'StandaloneBuildingBlock') -> None:
        """Change out specified blocks with new block"""
        blocks = [self.blocks[i] for i in block_ids]
        assert all(block.connections == new_block.connections for block in blocks)
        dangling_bonds_list = [DanglingBond.from_block(b, invert=True) for b in blocks]
        self.remove_blocks(block_ids)
        for dangling_bonds in dangling_bonds_list:
            self.insert_block(new_block, dangling_bonds)
        return

    def rotate_blocks(self, block_ids: Iterable[int], angle: float) -> None:
        """"""
        blocks = [self.blocks[i] for i in block_ids]
        assert all(block.connections == 2 for block in blocks)
        for block in blocks:
            pos = [bond.pos[1] for bond in DanglingBond.from_block(block)]
            dummy = ase.Atoms(numbers=[0] * len(block), positions=block.positions)
            dummy.rotate(angle, pos[1] - pos[0], center=pos[0])
            block.set_positions(dummy.positions)
        return

    def _make_ghost_atoms(self) -> None:
        """"""
        if not len(self.dangling_bonds):
            self.ghost_atoms = ase.Atoms()
            return

        pos = [bond.pos[1] for bond in self.dangling_bonds]
        self.ghost_atoms = ase.Atoms(numbers=np.zeros(len(pos)), positions=pos)

    def _make_termination(self) -> None:
        """"""
        if not len(self.dangling_bonds):
            self.termination = ase.Atoms()
            return

        pos_in, pos_out = [b.pos[0] for b in self.dangling_bonds], [b.pos[1] for b in self.dangling_bonds]
        self.termination = make_termination(self.atoms, np.array(pos_in), np.array(pos_out))
        self.termination.arrays |= {'ref_idx': - np.ones(len(pos_in))}


class StandaloneBuildingBlock:
    """Unbreakable group of atoms"""

    def __init__(self, atoms: ase.Atoms, graph: nx.Graph, dangling_bonds: list[DanglingBond]):
        self.atoms, self.graph, self.dangling_bonds = atoms, graph, dangling_bonds
        self.atoms.cell = None

    @classmethod
    def from_block(cls, block: BuildingBlock):
        block.mic()  # do not cross periodic boundaries
        graph, atoms, dangling_bonds = block.graph.copy(), block.get_atoms(), DanglingBond.from_block(block)
        return cls(atoms, graph, dangling_bonds)

    def move_to(self, pos: np.ndarray) -> None:
        """"""
        self.atoms.positions += pos - self.pos

    def align_atoms(self, dangling_bonds: list[DanglingBond]) -> ase.Atoms:
        """"""
        assert len(dangling_bonds) == len(self.dangling_bonds)

        # dangling bonds point in the opposite direction
        pts = np.concatenate([bond.pos for bond in self.dangling_bonds])
        target_pts = np.concatenate([bond.pos[::-1] for bond in dangling_bonds])
        scale_factor = compute_inertia(target_pts) / compute_inertia(pts)
        transform, *_ = match_point_clouds(target_pts, pts * scale_factor)

        atoms = self.get_atoms()
        atoms.positions = transform(atoms.positions * scale_factor)
        return atoms

    # def merge(self, other: 'BuildingBlock') -> list[tuple[int, int]]:
    #     """Merge blocks and return bonds that were recombined"""
    #     # TODO: more sanity checks?
    #     if not isinstance(other, BuildingBlock) or not (self.template is other.template):
    #         raise AssertionError('You can only combine blocks of a single template..')
    #
    #     labels = sorted(self.labels + other.labels)
    #     graph = nx.compose(self.graph, other.graph)
    #     broken_bonds = self.broken_bonds.union(other.broken_bonds)
    #     new_bonds = [b for b in broken_bonds if b[::-1] in broken_bonds]        # recombined bonds
    #     if not new_bonds:
    #         print('Combining two blocks that do not share any bonds..')         # TODO: warning?
    #     graph.add_edges_from(new_bonds)
    #     self.labels, self.graph = labels, graph
    #     self.reset()
    #     return new_bonds

    def get_atoms(self, ghost_atoms: bool = False) -> ase.Atoms:
        # TODO: duplicated code..?
        atoms = self.atoms.copy()
        if ghost_atoms:
            pos = [bond.pos[1] for bond in self.dangling_bonds]
            atoms += ase.Atoms(numbers=np.zeros(len(pos)), positions=pos)
        return atoms

    @property
    def bonds(self) -> list[tuple[int, int]]: return list(self.graph.edges)

    @property
    def pos(self) -> np.ndarray: return self.positions.mean(axis=0)

    @property
    def positions(self) -> np.ndarray: return self.atoms.positions

    @property
    def formula(self) -> dict: return self.atoms.symbols.formula

    @property
    def connections(self) -> int: return len(self.dangling_bonds)

    def __len__(self) -> int: return len(self.atoms)

    def __str__(self):
        return f'BuildingBlock({self.formula}, pos: {[round(_, 2) for _ in self.pos]}, connections: {self.connections})'
