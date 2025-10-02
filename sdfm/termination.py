from typing import Sequence

import ase
import ase.geometry
import numpy as np

BOND_CUTOFF = 4  # TODO: arbitrary


def make_termination(atoms: ase.Atoms, pos_in: np.ndarray, pos_out: np.ndarray) -> ase.Atoms:
    """Create termination at dangling bonds - currently simple H atoms along the bond vector"""
    raw_vecs = pos_out - pos_in
    bond_vecs, bond_dists = ase.geometry.find_mic(raw_vecs, cell=atoms.cell, pbc=atoms.pbc)
    assert not np.any(bond_dists > BOND_CUTOFF), f'Found bond distance larger than {BOND_CUTOFF} angstroms..'
    vecs = bond_vecs / bond_dists[:, None]

    # terminate with simple hydrogens, for now
    termination = ase.Atoms(numbers=np.ones(pos_in.shape[0]), positions=pos_in + 1.1 * vecs)

    return termination
