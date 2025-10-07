from typing import Iterable
from functools import cached_property

import ase
import ase.geometry
import numpy as np
import networkx as nx
from networkx.algorithms.isomorphism import vf2pp_is_isomorphic

from sdfm.utils import sort_objects

KEY_TEMPLATE = 'template_name'
BOND_CUTOFF = 5  # TODO: arbitrary


def find_aromatic_rings(graph: nx.Graph) -> list[nx.Graph]:
    """"""
    # cyclical graphs of 6 carbons
    graph_carbon = graph.subgraph([l for (l, n) in graph.nodes.data('number') if n == 6])
    cycles = list(nx.simple_cycles(graph_carbon, length_bound=6))
    rings = [graph.subgraph(c) for c in cycles if len(c) == 6]

    # isomorphic to aromatic ring       TODO: required?
    ring_template = nx.Graph()
    ring_template.add_nodes_from([(i, {'number': 6}) for i in range(6)])
    ring_template.add_edges_from([(i, (i + 1) % 6) for i in range(6)])
    return [r for r in rings if vf2pp_is_isomorphic(r, ring_template)]


def graph_from_atoms(atoms: ase.Atoms) -> nx.Graph:
    """"""
    from ase.neighborlist import build_neighbor_list, NewPrimitiveNeighborList
    # TODO: keys should not be strings
    # TODO: mark more bonds as (un)breakable

    # find bonds -- this is still quite suboptimal -- move to faster neighbourlists if necessary
    nbl = build_neighbor_list(atoms, self_interaction=False, primitive=NewPrimitiveNeighborList)
    bonds = np.stack(nbl.get_connectivity_matrix().nonzero()).T

    graph = nx.Graph()
    graph.add_nodes_from([(idx, {'number': number}) for idx, number in enumerate(atoms.numbers)])
    graph.add_edges_from(bonds)

    # do not break rings, do break connections to rings
    aromatic_rings = find_aromatic_rings(graph)
    for ring in aromatic_rings:
        [graph.edges[e].update({'breakable': False, 'order': 1.5}) for e in ring.edges]
        [graph.nodes[n].update({'aromatic': True}) for n in ring.nodes]
        for e in nx.edge_boundary(graph, ring):
            graph.edges[e].setdefault('breakable', True)
            graph.edges[e].setdefault('order', 1)

    # make sure atoms and graph nodes are labelled consistently
    if 'labels' in atoms.arrays:
        node_map = {i: l for i, l in enumerate(atoms.arrays['labels'])}
        graph = nx.relabel_nodes(graph, node_map)
    else:
        atoms.arrays['labels'] = np.arange(len(atoms))

    return graph


def components_from_graph(graph: nx.Graph) -> list[set[int]]:
    """"""
    bonds_to_break = [e[:2] for e in graph.edges.data('breakable') if e[-1]]
    (graph := graph.copy()).remove_edges_from(bonds_to_break)
    return [labels for labels in nx.connected_components(graph)]


def coarse_grain_graph(graph: nx.Graph, blocks: list['BuildingBlock']) -> nx.Graph:
    """Make a coarse-grained representation of a graph by condensing nodes within each block to single nodes.
    New edges are defined between all blocks that share an edge."""
    label_to_block = {l: b for b in blocks for l in b.labels}
    edges_cg = [(b, label_to_block[l]) for b in blocks for l in nx.node_boundary(graph, b.labels)]
    graph = nx.Graph(edges_cg)
    graph.add_nodes_from(blocks)  # include unconnected blocks
    return graph


class AtomsTemplate:
    """Container for a molecular system template"""
    # TODO: does having all these attributes as property make sense?
    graph: nx.Graph
    labels: np.ndarray
    graph_cg: nx.Graph
    blocks: list['BuildingBlock']

    def __init__(self, atoms: ase.Atoms, name: str = None, graph: nx.Graph = None):
        """"""
        self.atoms, self.graph = atoms.copy(), graph
        self.blocks = self.graph_cg = None
        if name:
            self.name = atoms.info[KEY_TEMPLATE] = name
        else:
            self.name = atoms.info.get(KEY_TEMPLATE)
        self.init()

    def init(self) -> None:
        """"""
        if self.graph is None:
            self.atoms.arrays['labels'] = self._make_node_labels(len(self))
            self.graph = graph_from_atoms(self.atoms)
        else:
            # check whether ids <> labels mapping is consistent
            assert len(self.graph) == len(self.atoms) and set(self.labels) == set(self.graph), 'What in tarnation'
        block_labels = components_from_graph(self.graph)
        self.blocks = sorted([BuildingBlock(self, l) for l in block_labels], key=len, reverse=True)
        self.graph_cg = coarse_grain_graph(self.graph, self.blocks)

    def merge_blocks(self, min_size: int) -> None:
        """Combine blocks until all have a minimal size"""
        # TODO: we currently blacklist blocks that have insufficient neighbours to continue merging.. this is not good..?
        current_size = len(self.blocks[-1])
        ids_blacklist = set()
        while len(self.blocks) > 1 and current_size < min_size:
            try:
                # find first block of smallest size
                idx, block = next((idx, block) for idx, block in enumerate(self.blocks) if
                                  len(block) == current_size and idx not in ids_blacklist)
            except StopIteration:
                # none found -> increase smallest size
                current_size = min(len(block) for idx, block in enumerate(self.blocks) if idx not in ids_blacklist)
                continue

            # merge block with smallest neighbour
            neighbours = sorted(nx.node_boundary(self.graph_cg, [block]), key=len)
            if len(neighbours) == 0:
                # TODO: what when this happens?
                ids_blacklist.add(idx)
                continue
            neighbour = neighbours.pop(0)
            recombined_edges = neighbour.merge(block)

            # update template
            self.graph_cg.remove_node(self.blocks.pop(idx))
            self.graph_cg.add_edges_from([(neighbour, other) for other in neighbours])
            for e in recombined_edges:
                self.graph.edges[e]['breakable'] = False

        self.blocks = sorted(self.blocks, key=len, reverse=True)
        for block in self.blocks:
            block.mic()  # make sure new blocks do not cross periodic boundaries
        self._check_sanity()

    def make_supercell(self, x: int, y: int, z: int) -> 'AtomsTemplate':
        """"""
        # make supercell and map new to old atoms
        atoms_new = self.atoms.repeat((x, y, z))
        multi = int(len(atoms_new) / len(self.atoms))
        labels = atoms_new.arrays.pop('labels')
        atoms_new.arrays['labels'] = labels_new = self._make_node_labels(len(atoms_new))
        map_labels = sort_objects(labels_new, filter_keys=labels)

        nodes_new = []
        for k, v in map_labels.items():
            kwargs = self.graph.nodes[k]
            nodes_new += [(i, kwargs) for i in v]

        edges_new = []
        for e in self.graph.edges:
            kwargs = self.graph.edges[e]
            edge_pos = self.atoms.positions[list(e)]
            edge_across_pbc = np.square(edge_pos[0] - edge_pos[1]).sum() > BOND_CUTOFF ** 2

            edge_nodes_1, edge_nodes_2 = map_labels[e[0]], map_labels[e[1]]
            if not edge_across_pbc:
                # no reordering, simply keep bond indices for periodic copies
                edges_new += [(edge_nodes_1[_], edge_nodes_2[_], kwargs) for _ in range(multi)]
            else:
                # reordering, find out which atoms are connected
                edge_pos = atoms_new.positions[[edge_nodes_1, edge_nodes_2]]
                edge_vecs = edge_pos[0, :, None] - edge_pos[1, None, :]

                # find non-pbc bonds
                edge_mask = np.linalg.norm(edge_vecs, axis=-1) < BOND_CUTOFF
                edges_new += [(edge_nodes_1[i], edge_nodes_2[j], kwargs) for (i, j) in np.argwhere(edge_mask)]

                # find remaining pbc bonds
                # slow because 'general_find_mic' wants to test all 27 first periodic neighbours
                edge_mask_pbc = ~edge_mask.any(axis=1)[:, None] * ~edge_mask.any(axis=0)[None]
                edge_vecs_pbc, edge_ids_pbc = edge_vecs[edge_mask_pbc], np.argwhere(edge_mask_pbc)
                _, edge_norms_pbc = ase.geometry.find_mic(edge_vecs_pbc, cell=atoms_new.cell, pbc=atoms_new.pbc)
                edge_ids = edge_ids_pbc[edge_norms_pbc < BOND_CUTOFF]
                edges_new += [(edge_nodes_1[i], edge_nodes_2[j], kwargs) for (i, j) in edge_ids]

        graph = nx.Graph()
        graph.add_nodes_from(nodes_new)
        graph.add_edges_from(edges_new)

        pos = atoms_new.positions[graph.edges]
        _, norm = ase.geometry.find_mic(pos[:, 0] - pos[:, 1], cell=atoms_new.cell, pbc=atoms_new.pbc)
        assert (len(self.graph) * multi == len(graph) and
                len(self.graph.edges) * multi == len(graph.edges) and
                np.all(norm < BOND_CUTOFF))

        return self.__class__(atoms=atoms_new, name=self.name, graph=graph)

    def labels_to_ids(self, labels: list[int]) -> list[int]:
        return labels

    def _make_node_labels(self, size: int, start: int = 0) -> np.ndarray:
        return np.arange(size) + start

    def _check_sanity(self) -> None:
        """"""
        # TODO: bond length check?
        labels = sum([b.labels for b in self.blocks], [])
        assert (
                (len(self.labels) == len(self.atoms)) *
                (self.graph.number_of_nodes() == len(self.atoms)) *
                (self.graph_cg.number_of_nodes() == len(self.blocks)) *
                (len(labels) == len(set(labels)) == len(self.atoms))
        ), 'This template is broken..'

    def __len__(self) -> int:
        return self.atoms.__len__()

    def __str__(self) -> str:
        return (f'AtomsTemplate(name: {self.name}, atoms: {self.__len__()}, bonds: {self.graph.number_of_edges()}, '
                f'blocks: {len(self.blocks)}, connections: {self.graph_cg.number_of_edges()})')

    @property
    def bonds(self) -> list[tuple[int, int]]:
        """All bonds in graph"""
        return list(self.graph.edges)

    @property
    def connections(self) -> list[tuple[int, int]]:
        """All bonds in graph that can be broken"""
        return [b[:2] for b in self.graph.edges(data='breakable') if b[-1]]

    @property
    def block_ids(self) -> list[list[int]]:
        """Partitioning of atomic indices into building blocks"""
        return [block.ids for block in self.blocks]

    @property
    def block_labels(self) -> list[list[int]]:
        """Partitioning of atomic labels into building blocks"""
        return [block.labels for block in self.blocks]

    @property
    def bonds_cg(self) -> list[tuple[int, int]]:
        """All bonds in CG graph, i.e., every connection between building blocks"""
        return list(self.graph_cg.edges)

    @property
    def pos(self) -> np.ndarray:
        return self.atoms.positions

    @property
    def cell(self) -> ase.atoms.Cell:
        return self.atoms.cell

    @property
    def pbc(self) -> np.ndarray:
        return self.atoms.pbc

    @property
    def labels(self) -> np.ndarray:
        return self.atoms.arrays['labels']


class BuildingBlock:
    """Unbreakable group of atoms within an AtomsTemplate"""

    # TODO: maybe keep track of where bonds are broken? Shortest path to any broken bond might be useful information?
    # TODO: storing functionality -> reinit with dummy template?
    # TODO: have neighbours as a property?
    def __init__(self, template: AtomsTemplate, labels: Iterable[int]):
        self.template, self.labels = template, sorted(labels)
        self.graph = template.graph.subgraph(labels)
        self.mic()

    def mic(self) -> None:
        """Shift atoms of this building block to eliminate periodic boundary crossings"""
        center = (cell := self.parent.cell).scaled_positions((pos := self.positions)[0])
        self.set_positions(ase.geometry.wrap_positions(pos, cell, center=center))

    def move_to(self, pos: np.ndarray) -> None:
        """"""
        self.set_positions(self.positions + pos - self.pos)

    def merge(self, other: 'BuildingBlock') -> list[tuple[int, int]]:
        """Merge blocks and return bonds that were recombined"""
        # TODO: more sanity checks?
        if not isinstance(other, BuildingBlock) or not (self.template is other.template):
            raise AssertionError('You can only combine blocks of a single template..')

        labels = sorted(self.labels + other.labels)
        graph = nx.compose(self.graph, other.graph)
        broken_bonds = self.broken_bonds.union(other.broken_bonds)
        new_bonds = [b for b in broken_bonds if b[::-1] in broken_bonds]  # recombined bonds
        if not new_bonds:
            print('Combining two blocks that do not share any bonds..')  # TODO: warning?
        graph.add_edges_from(new_bonds)
        self.labels, self.graph = labels, graph
        self.reset()
        return new_bonds

    def get_atoms(self) -> ase.Atoms:
        at = self.parent[self.ids]
        at.pbc = False
        return at

    def set_positions(self, pos: np.ndarray) -> None:
        self.parent.positions[self.ids] = pos

    def reset(self) -> None:
        self.__dict__.pop('ids', None)
        self.__dict__.pop('broken_bonds', None)
        self.__dict__.pop('formula', None)

    @cached_property
    def ids(self):
        return self.template.labels_to_ids(self.labels)

    @cached_property
    def broken_bonds(self) -> set[tuple[int, int]]:
        return set(nx.edge_boundary(self.template.graph, self.labels))

    @property
    def parent(self):
        return self.template.atoms

    @property
    def bonds(self) -> list[tuple[int, int]]:
        return list(self.graph.edges)

    @property
    def pos(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    @property
    def positions(self) -> np.ndarray:
        return self.parent.positions[self.ids]

    @cached_property
    def formula(self) -> dict:
        return self.parent.symbols[self.ids].formula  # TODO: maybe not cached?

    @property
    def numbers(self) -> np.ndarray:
        return self.parent.numbers[self.ids]

    @property
    def connections(self) -> int:
        return len(self.broken_bonds)

    def __len__(self) -> int:
        return len(self.labels)

    def __str__(self):
        return f'BuildingBlock({self.formula}, pos: {[round(_, 2) for _ in self.pos]}, connections: {self.connections})'


def is_matching_template(template: AtomsTemplate, atoms: ase.Atoms) -> bool:
    """TODO: not foolproof"""
    if (
            len(atoms) != len(template) or
            not (atoms.numbers == template.atoms.numbers).all() or
            not (atoms.pbc == template.pbc).all()
    ):
        return False

    bond_labels = template.bonds
    bond_indices = [template.labels_to_ids(bond) for bond in bond_labels]
    bond_pos = atoms.positions[bond_indices]
    _, bond_dists = ase.geometry.find_mic(bond_pos[:, 0] - bond_pos[:, 1], atoms.cell, atoms.pbc)
    return (bond_dists <= BOND_CUTOFF).all()
