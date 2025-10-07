import subprocess
import tempfile
import tkinter as tk

import ase
import numpy as np
import ase.io
import ase.visualize
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ase.data import atomic_names as names
from ase.data import chemical_symbols as symbols
from ase.gui.gui import GUI

from sdfm.template import AtomsTemplate
from sdfm.modify import ExtendedAtomsTemplate, DanglingBond
from sdfm.selection import AtomSelection

VESTA_COMMAND = 'vesta'


def modify_gui(gui: GUI, keep_menu: bool = False, keep_status: bool = False, foreground: bool = True) -> None:
    """"""
    root = gui.window.win
    if not keep_menu:
        root.children['!menu'].destroy()  # remove ASE menu
    if not keep_status:
        root.children['!label'].pack_forget()  # disable ASE selection label
    if foreground:
        root.attributes('-topmost', 'true')


def status_blocks(self: GUI, atoms: ase.Atoms):
    """"""
    ids = np.flatnonzero(self.images.selected)
    if not ids.size == 1:
        self.window.update_status_line('')
        return

    idx = ids[0]
    number, pos = atoms.numbers[idx], atoms.positions[idx]
    name, symbol = names[number], symbols[number]
    text = f'#{idx} {name} ({symbol}): ' + '{:.3f} Å, {:.3f} Å, {:.3f} Å'.format(*pos)
    if 'block_ids' in atoms.arrays:
        text += ' - block={}'.format(atoms.arrays['block_ids'][idx])

    self.window.update_status_line(text)


def visualise_template_blocks(template: AtomsTemplate, cmap: str = 'tab10'):
    """"""
    atoms = template.atoms.copy()
    colours = [colors.rgb2hex(c[:3]) for c in plt.cm.get_cmap(cmap).colors]
    ids, block_ids = np.zeros_like(atoms.numbers), np.zeros_like(atoms.numbers)
    for idx, block in enumerate(template.blocks):
        ids[block.ids] = idx % len(colours)
        block_ids[block.ids] = idx
    atoms.arrays.update({'colours': ids, 'block_ids': block_ids})

    gui = GUI(images=[atoms])
    gui.status = status_blocks.__get__(gui, GUI)  # overwrite function
    gui.colormode, gui.colormode_data = 'colours', (colours, 0, len(colours))
    modify_gui(gui, keep_status=True)
    gui.draw()
    tk.mainloop()
    return


def visualise_dangling_bonds(dangling_bonds: list[DanglingBond]) -> None:
    """"""
    pos = np.stack([bond.pos for bond in dangling_bonds]).reshape(-1, 3)
    numbers = [1, 0] * len(dangling_bonds)
    atoms = ase.Atoms(positions=pos, numbers=numbers)
    ase.visualize.view(atoms)


def visualise_selection(selection: AtomSelection) -> None:
    """"""
    atoms, mask = selection._template.atoms.copy(), selection.get_atoms_mask()
    gui = GUI(images=[atoms])
    modify_gui(gui, keep_status=True)
    gui.images.selected[:] = mask
    gui.b1 = None  # unset mouse left click
    gui.draw()
    gui.window.win.mainloop()


# TODO: everything below is old

class AtomsGUI:
    """"""

    # TODO: to general utils?
    def __init__(self, atoms: ase.Atoms):
        self.gui = GUI()
        self.root.children['!menu'].destroy()  # remove ASE menu
        self.root.children['!label'].pack_forget()  # disable ASE selection label
        self.root.attributes('-topmost', 'true')
        self.set_atoms(atoms)
        self.draw(focus=True)

        # self.gui.run()
        # self.gui.repeat_poll(self.update, 100)
        # self.root.after()

    @property
    def atoms(self): return self.gui.atoms

    @property
    def images(self): return self.gui.images

    @property
    def root(self): return self.gui.window.win

    def set_atoms(self, atoms: ase.Atoms):
        """"""
        self.images.initialize([atoms])

    def draw(self, focus: bool = False):
        self.root.update()
        self.gui.set_frame(0, focus=focus)

    def stop(self) -> None:
        self.root.destroy()


class OverviewWindow(tk.Frame):
    """Store and display some atomic information"""

    def __init__(self, master):
        super().__init__(master=master, name='overview')
        kwargs = dict(master=self, font=('latin modern roman', 12, 'bold'))
        tk.Label(text='OVERVIEW', **kwargs).grid(row=0, columnspan=3)
        tk.Label(text='STRUCTURE', **kwargs).grid(row=1, column=0)
        tk.Label(text='SELECTION', **kwargs).grid(row=1, column=1)
        tk.Label(text='VIEW', **kwargs).grid(row=1, column=2)
        kwargs = dict(master=self, height=4, width=20)
        tk.Text(name='structure', **kwargs).grid(row=2, column=0)
        tk.Text(name='selection', **kwargs).grid(row=2, column=1)
        tk.Text(name='view', **kwargs).grid(row=2, column=2)
        tk.Label(master=self, text='').grid(row=3, columnspan=3)

    def update_fields(self, atoms: int, blocks: int, dangling: int, formula: str,
                      level: str, selected: int, sformula: str, frame: int) -> None:
        """"""
        text = '\n'.join([f'atoms: {atoms}', f'blocks: {blocks}', f'dangling: {dangling}', formula])
        self._set_text_widget(text, widget_name='structure')
        text = '\n'.join([f'level: {level}', f'selected: {selected}', '', sformula])
        self._set_text_widget(text, widget_name='selection')
        self._set_text_widget(f'frame: {frame}', widget_name='view')
        self.update()

    def _set_text_widget(self, text: str, widget: tk.Text = None, widget_name: str = None):
        if widget is None:
            widget = self.children[widget_name]
        widget.delete(1.0, tk.END)
        widget.insert(1.0, chars=text)


class TemplateGUI(AtomsGUI):
    """See things interactively yeey"""

    def __init__(self, template: ExtendedAtomsTemplate, selection: AtomSelection):
        super().__init__(template.get_atoms(wrap=True))
        self.template = template
        self.overview = OverviewWindow(self.root)
        self.overview.pack(side=tk.TOP, expand=True)

        self.settings = {'wrap': True, 'ghost_atoms': True, 'termination': False}
        self.frame = 0
        self.update(selection)

    def update(self, selection: AtomSelection, **kwargs):
        """"""
        self.settings |= kwargs
        atoms = self.template.get_atoms(**self.settings)
        if not selection.is_consistent:
            print('Selection is no longer consistent with template. Reinitialise..')
            selection = AtomSelection(self.template, level=selection.level)
        atoms_mask = selection.get_mask()
        self.overview.update_fields(
            atoms=selection.n_atoms,
            blocks=selection.n_blocks,
            dangling=len(self.template.dangling_bonds),
            formula=atoms.symbols.get_chemical_formula(),
            level=selection.level,
            selected=selection.n_selected,
            sformula=atoms.symbols[:atoms_mask.size][atoms_mask].get_chemical_formula(),
            frame=self.frame,
        )
        self.set_atoms(atoms)
        self.images.selected[:len(atoms_mask)] = atoms_mask
        self.draw(focus=True)
        self.frame += 1

    def open_in_vesta(self, vesta_command: str = VESTA_COMMAND):
        with tempfile.NamedTemporaryFile(mode='a', suffix='.cif') as f:
            ase.io.write(f.name, self.gui.atoms, format='cif')
            subprocess.run(args=[vesta_command, f.name])

    def open_in_ase(self):
        ase.visualize.view(self.gui.atoms)
