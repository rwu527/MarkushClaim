
import io
import os
import random
from tkinter import Image
from rdkit import Chem
from rdkit.Chem import rdmolops
from PIL import Image
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DCairo
from src.fragment import fragment_and_draw, sanitize_filename
from src import global_counters



def mark_mcs_with_r(molecule_list, mcs_smiles, output_folder, differences):

    mcs_mol = Chem.MolFromSmarts(mcs_smiles)
    if not mcs_mol:
        return None, None, None, None, None

    r_group_counts = {idx: 0 for idx in range(mcs_mol.GetNumAtoms())}
    mcs_with_r = Chem.RWMol(mcs_mol)
    r_group_mapping = {}
    x_group_mapping = {}
    z_group_mapping = {}
    bond_indices = []
    atom_indices = set()
    non_carbon_atom_replacements = {}
    z_group_replacements = {}

    for mol in molecule_list:
        match = mol.GetSubstructMatch(mcs_mol)
        r_group_temp_counts = {idx: 0 for idx in range(mcs_mol.GetNumAtoms())}
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx not in match:
                for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in match:
                        r_atom_idx = match.index(neighbor_idx)
                        r_group_temp_counts[r_atom_idx] += 1
        for idx in r_group_temp_counts:
            r_group_counts[idx] = max(r_group_counts[idx], r_group_temp_counts[idx])
            
    for r_atom_idx, count in r_group_counts.items():
        for _ in range(count):
            if mcs_with_r.GetAtomWithIdx(r_atom_idx).GetDegree() == 1:
                if r_atom_idx not in r_group_mapping:
                    r_group_mapping[r_atom_idx] = f'R{str(global_counters.global_r_group_counter)}'
                    global_counters.global_r_group_counter += 1
                r_group_atom = mcs_with_r.GetAtomWithIdx(r_atom_idx)
                r_group_atom.SetProp('atomLabel', r_group_mapping[r_atom_idx])
                for neighbor in r_group_atom.GetNeighbors():
                    atom_indices.add(neighbor.GetIdx())
            else:
                r_group_atom = Chem.Atom(0)
                r_group_idx = mcs_with_r.AddAtom(r_group_atom)
                mcs_with_r.AddBond(r_atom_idx, r_group_idx, Chem.BondType.SINGLE)
                r_group_mapping[r_group_idx] = f'R{(str(global_counters.global_r_group_counter))}'
                mcs_with_r.GetAtomWithIdx(r_group_idx).SetProp('atomLabel', r_group_mapping[r_group_idx])
                atom_indices.add(r_atom_idx)
                global_counters.global_r_group_counter += 1

    for atom_idx in range(mcs_with_r.GetNumAtoms()):
        atom = mcs_with_r.GetAtomWithIdx(atom_idx)
        atom_symbol = atom.GetSymbol()

    rings = rdmolops.GetSymmSSSR(mcs_with_r)

    for ring in rings:
        for atom_idx in ring:
            atom = mcs_with_r.GetAtomWithIdx(atom_idx)
            atom_symbol = atom.GetSymbol()
            if atom_symbol != 'C':
                found_non_carbon_in_rings = True
                if atom_idx not in r_group_mapping:  # Exclude atoms already labeled as R
                    if atom_idx not in x_group_mapping:  # Prevent duplicate labeling
                        x_group_mapping[atom_idx] = f'X{(str(global_counters.global_x_group_counter))}'
                        global_counters.global_x_group_counter += 1
                    mcs_with_r.ReplaceAtom(atom_idx, Chem.Atom(0))
                    x_group_atom = mcs_with_r.GetAtomWithIdx(atom_idx)
                    x_group_atom.SetProp('atomLabel', x_group_mapping[atom_idx])
                    if x_group_mapping[atom_idx] not in non_carbon_atom_replacements:
                        non_carbon_atom_replacements[x_group_mapping[atom_idx]] = []
                    non_carbon_atom_replacements[x_group_mapping[atom_idx]].append(atom_symbol)

    # Handle remaining atoms and bonds
    for atom_idx in range(mcs_with_r.GetNumAtoms()):
        atom = mcs_with_r.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() not in ['C', 'H']:
            if atom.GetDegree() == 1 and len(atom.GetNeighbors()) == 1:
                neighbor_idx = atom.GetNeighbors()[0].GetIdx()
                if atom_idx not in r_group_mapping:  # Ensure it's not an R atom
                    if atom_idx not in z_group_mapping:  # Prevent duplicate labeling
                        z_group_mapping[atom_idx] = f'Z{(str(global_counters.global_z_group_counter))}'
                        global_counters.global_z_group_counter += 1
                    atom.SetProp('atomLabel', z_group_mapping[atom_idx])
                    if z_group_mapping[atom_idx] not in z_group_replacements:
                        z_group_replacements[z_group_mapping[atom_idx]] = []
                    z_group_replacements[z_group_mapping[atom_idx]].append(atom.GetSymbol())
                       
    # Handle bonds
    for bond in mcs_with_r.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if (begin_idx in atom_indices or end_idx in atom_indices) and not (begin_idx in r_group_mapping or end_idx in r_group_mapping):
            bond_indices.append((begin_idx, end_idx))

    atom_r_mapping = {}
    for atom_idx in atom_indices:
        connected_r_groups = []
        for neighbor in mcs_with_r.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in r_group_mapping:
                connected_r_groups.append(r_group_mapping[neighbor_idx])
        atom_r_mapping[atom_idx] = connected_r_groups

    original_atom_mappings = map_indices_to_original(molecule_list, mcs_mol, atom_r_mapping)
    original_bond_mappings = map_bonds_to_original(molecule_list, mcs_mol, bond_indices)
    
    # Check N atoms and label H
    for atom_idx in range(mcs_with_r.GetNumAtoms()):
        atom = mcs_with_r.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'N' and atom.GetDegree() == 2:
            single_bonds = sum(1 for bond in atom.GetBonds() if bond.GetBondType() == Chem.BondType.SINGLE)
            
            if single_bonds == 2:  # If it is a -N- structure
                # Directly label hydrogen atom without adding actual bonds
                atom.SetProp('atomLabel', 'NH')  # Set nitrogen atom label to 'NH'

    # Now generate output for X and Z groups
    x_groups_created = set()  # Used to track created X group folders
    for x_group, replaced_atoms in non_carbon_atom_replacements.items():
        if not replaced_atoms:
            del x_group_mapping[x_group]
        else:
            create_x_group_folder(output_folder, x_group, replaced_atoms)
            x_groups_created.add(x_group)

    z_groups_created = set()  # Used to track created Z group folders
    for z_group, replaced_atoms in z_group_replacements.items():
        if not replaced_atoms:
            del z_group_mapping[z_group]
        else:
            create_z_group_folder(output_folder, z_group, replaced_atoms)
            z_groups_created.add(z_group)

    return mcs_with_r, list(atom_indices), bond_indices, original_atom_mappings, original_bond_mappings


def create_z_group_folder(output_folder, z_group, replaced_atoms):
    z_group_folder = os.path.join(output_folder, z_group)
    os.makedirs(z_group_folder, exist_ok=True)
    z_group_file = os.path.join(z_group_folder, 'replacements.txt')
    with open(z_group_file, 'w') as f:
        for atom in replaced_atoms:
            f.write(f'{atom}\n')


def create_x_group_folder(output_folder, x_group, replaced_atoms):
    x_group_folder = os.path.join(output_folder, x_group)
    os.makedirs(x_group_folder, exist_ok=True)
    x_group_file = os.path.join(x_group_folder, 'replacements.txt')
    with open(x_group_file, 'w') as f:
        for atom in replaced_atoms:
            f.write(f'{atom}\n')


def map_indices_to_original(molecule_list, mcs_mol, atom_r_mapping):
    original_mappings = []
    for mol in molecule_list:
        match = mol.GetSubstructMatch(mcs_mol)
        mol_mapping = {}
        for mcs_atom_idx, r_groups in atom_r_mapping.items():
            if mcs_atom_idx < len(match):
                original_atom_idx = match[mcs_atom_idx]
                mol_mapping[original_atom_idx] = r_groups
        original_mappings.append(mol_mapping)
    return original_mappings


def map_bonds_to_original(molecule_list, mcs_mol, bond_indices):
    original_bond_mappings = []
    for mol in molecule_list:
        match = mol.GetSubstructMatch(mcs_mol)
        bond_mapping = []
        if match: 
            for begin_idx, end_idx in bond_indices:
                if begin_idx < len(match) and end_idx < len(match):
                    original_begin_idx = match[begin_idx]
                    original_end_idx = match[end_idx]
                    original_bond = mol.GetBondBetweenAtoms(original_begin_idx, original_end_idx)
                    if original_bond:
                        original_bond_idx = original_bond.GetIdx()
                        bond_mapping.append((original_begin_idx, original_end_idx, original_bond_idx))
        original_bond_mappings.append(bond_mapping)
    return original_bond_mappings


def mark_nest_mcs_with_r(mcs_mapping, output_folder):
    initial_r_group_counter = global_counters.global_r_group_counter
    max_r_in_mcs = initial_r_group_counter
    marked_r_groups = []
    valid_molecules = []
    for i, (mcs_smarts, mcs_data) in enumerate(mcs_mapping.items(), start=1):
        global_counters.global_r_group_counter = initial_r_group_counter
        mcs_smiles = mcs_data['mcs_structure']
        fragments = mcs_data['fragments']
        mcs_mol = Chem.MolFromSmiles(mcs_smiles)

        if not mcs_mol:
            continue

        molecule_index = []  
        molecule_list = []
        for frag in fragments:
            frag_smiles = frag['smiles']  
            frag_file_name = frag['file_name'] 
            frag_mol = Chem.MolFromSmiles(frag_smiles)
            if frag_mol:
                molecule_list.append(frag_mol)
                fragment_index = int(frag_file_name.split('_')[1])  
                molecule_index.append((fragment_index, frag_mol)) 

        r_group_counts = {idx: 0 for idx in range(mcs_mol.GetNumAtoms())}
        r_group_mapping = {}
        bond_indices = []
        atom_indices = set()

        for mol in molecule_list:
            match = mol.GetSubstructMatch(mcs_mol)
            r_group_temp_counts = {idx: 0 for idx in range(mcs_mol.GetNumAtoms())}
            for atom_idx in range(mol.GetNumAtoms()):
                if atom_idx not in match:
                    for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()
                        if neighbor_idx in match:
                            r_atom_idx = match.index(neighbor_idx)
                            r_group_temp_counts[r_atom_idx] += 1
            for idx in r_group_temp_counts:
                r_group_counts[idx] = max(r_group_counts[idx], r_group_temp_counts[idx])

        mcs_with_r = Chem.RWMol(mcs_mol)
        atom_replacements = []
        for atom in mcs_with_r.GetAtoms():
            if atom.GetSymbol() == '*':
                atom_replacements.append(atom.GetIdx())
                atom.SetAtomicNum(1)

        try:
            smiles_before_kekulize = Chem.MolToSmiles(mcs_with_r, isomericSmiles=True)
            Chem.Kekulize(mcs_with_r, clearAromaticFlags=True)
        except Chem.rdchem.KekulizeException as e:
            print(f"Error during Kekulization of molecule: {e}")

        for idx in atom_replacements:
            mcs_with_r.GetAtomWithIdx(idx).SetAtomicNum(0)

        for r_atom_idx, count in r_group_counts.items():
            for _ in range(count):
                r_group_atom = mcs_with_r.GetAtomWithIdx(r_atom_idx)

                if r_group_atom.GetDegree() == 1:  
                    if r_atom_idx not in r_group_mapping:
                        r_group_mapping[r_atom_idx] = f'R{(str(global_counters.global_r_group_counter))}'
                        marked_r_groups.append(f'R{(str(global_counters.global_r_group_counter))}')
                        global_counters.global_r_group_counter += 1
                    r_group_atom.SetProp('atomLabel', r_group_mapping[r_atom_idx])
                    for neighbor in r_group_atom.GetNeighbors():
                        atom_indices.add(neighbor.GetIdx())
                else:
                    new_r_group_atom = Chem.Atom(0) 
                    new_r_group_idx = mcs_with_r.AddAtom(new_r_group_atom)
                    mcs_with_r.AddBond(r_atom_idx, new_r_group_idx, Chem.BondType.SINGLE)
                    r_group_mapping[new_r_group_idx] = f'R{(str(global_counters.global_r_group_counter))}'
                    marked_r_groups.append(f'R{(str(global_counters.global_r_group_counter))}')
                    mcs_with_r.GetAtomWithIdx(new_r_group_idx).SetProp('atomLabel', r_group_mapping[new_r_group_idx])
                    atom_indices.add(r_atom_idx)
                    global_counters.global_r_group_counter += 1
                    max_r_in_mcs = max(max_r_in_mcs, global_counters.global_r_group_counter)

                    atom_info = {
                        "Index": r_atom_idx,
                        "Symbol": r_group_atom.GetSymbol(),
                        "Degree": r_group_atom.GetDegree(),
                    }

                    if atom_info['Symbol'] in ['N', 'S', 'P', 'O']:
                        if r_group_atom.GetDegree() > 0:
                            new_atom = Chem.Atom(r_group_atom.GetAtomicNum())
                            mcs_with_r.ReplaceAtom(r_atom_idx, new_atom)

        for bond in mcs_with_r.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if (begin_idx in atom_indices or end_idx in atom_indices) and not (begin_idx in r_group_mapping or end_idx in r_group_mapping):
                bond_indices.append((begin_idx, end_idx))

        atom_r_mapping = {}
        for atom_idx in atom_indices:
            connected_r_groups = []
            for neighbor in mcs_with_r.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in r_group_mapping:
                    connected_r_groups.append(r_group_mapping[neighbor_idx])
            atom_r_mapping[atom_idx] = connected_r_groups

        original_atom_mappings = map_indices_to_original(molecule_list, mcs_mol, atom_r_mapping)
        original_bond_mappings = map_bonds_to_original(molecule_list, mcs_mol, bond_indices)

        if 'R' in mcs_smarts:
            continue
        
        # Here is the fix for properly generating and saving images
        asterisk_indices = [atom.GetIdx() for atom in mcs_with_r.GetAtoms() if atom.GetSymbol() == '*']
        r_indices = set(r_group_mapping.keys()) 
        true_asterisk_indices = [idx for idx in asterisk_indices if idx not in r_indices]  

        processed_asterisk_removal = False  
        
        if len(true_asterisk_indices) == 1:
            valid_molecules.append(mcs_with_r)
            
        if len(true_asterisk_indices) >= 2:
            try:
                # Phase 1: Record the non-asterisk end atom information of deleted double bonds
                retained_atom_mapnums = {}  # {original index: original mapping number}
                double_bond_pairs = []      # Record the non-asterisk end indices of cut double bonds

                for asterisk_idx in true_asterisk_indices:
                    atom = mcs_with_r.GetAtomWithIdx(asterisk_idx)
                    for bond in atom.GetBonds():
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            other_idx = bond.GetOtherAtomIdx(asterisk_idx)
                            if other_idx not in true_asterisk_indices:
                                other_atom = mcs_with_r.GetAtomWithIdx(other_idx)
                                original_mapnum = other_atom.GetAtomMapNum()
                                retained_atom_mapnums[other_idx] = original_mapnum
                                other_atom.SetAtomMapNum(other_idx)  # Temporary marking
                                double_bond_pairs.append(other_idx)

                # Phase 2: Perform the original deletion operation (unchanged)
                for idx in sorted(true_asterisk_indices, reverse=True):
                    bonds = list(mcs_with_r.GetAtomWithIdx(idx).GetBonds())
                    for bond in bonds:
                        mcs_with_r.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    mcs_with_r.RemoveAtom(idx)
                processed_asterisk_removal = True

                # Phase 3: Double bond repair logic
                if len(double_bond_pairs) >= 2:
                    retained_atoms = []
                    for atom in mcs_with_r.GetAtoms():
                        if atom.GetAtomMapNum() in double_bond_pairs:
                            original_idx = atom.GetAtomMapNum()
                            atom.SetAtomMapNum(retained_atom_mapnums[original_idx])  # Restore original mapping number
                            retained_atoms.append(atom.GetIdx())
                    
                    if len(retained_atoms) == 2:
                        a, b = retained_atoms
                        bond = mcs_with_r.GetBondBetweenAtoms(a, b)
                        if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                            backup_mol = Chem.Mol(mcs_with_r.ToBinary())
                            try:
                                bond.SetBondType(Chem.BondType.DOUBLE)
                                Chem.SanitizeMol(mcs_with_r)
                            except:
                                mcs_with_r = backup_mol

                for atom in mcs_with_r.GetAtoms():
                    atom.SetAtomMapNum(0)  # Reset mapping number to zero

            except Exception as e:
                processed_asterisk_removal = False
            
        AllChem.Compute2DCoords(mcs_with_r)
        drawer = Draw.MolDraw2DCairo(300, 300)
        options = drawer.drawOptions()
        options.useBWAtomPalette()
        options.colorAtoms = False
        options.highlightColor = None

        drawer.DrawMolecule(mcs_with_r)
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        img = img.convert("RGB")

        # Saving the image for the structure
        if processed_asterisk_removal:
            img_path = os.path.join(output_folder, f'0_mcs_with_r_{i}_{sanitize_filename(mcs_smarts)}.png')
        else:
            img_path = os.path.join(output_folder, f'mcs_with_r_{i}_{sanitize_filename(mcs_smarts)}.png')
        img.save(img_path)

        mcs_mapping[mcs_smarts]['r_group_mapping'] = r_group_mapping
        fragment_and_draw(molecule_list, molecule_index, original_atom_mappings, original_bond_mappings, output_folder)
        
    generated_smiles = extend_mcs_with_r(valid_molecules, output_folder)
    extend_mcs_with_ring(generated_smiles, output_folder)
    
    global_counters.global_r_group_counter = max_r_in_mcs
    return marked_r_groups


from copy import deepcopy


def extend_mcs_with_r(valid_molecules, output_folder):

    # List to store all generated molecules' SMILES to check for duplicates
    generated_smiles = set()
    
    def is_equivalent_to_existing(smiles, existing_smiles_set):
        mol1 = Chem.MolFromSmiles(smiles)
        if mol1 is None:
            return False
        smi1 = Chem.MolToSmiles(mol1, canonical=True)
        
        for ex_smiles in existing_smiles_set:
            mol2 = Chem.MolFromSmiles(ex_smiles)
            if mol2 is None:
                continue
            smi2 = Chem.MolToSmiles(mol2, canonical=True)
            if smi1 == smi2:
                return True
        return False
    
    for i, mcs_with_r in enumerate(valid_molecules):
        smiles = Chem.MolToSmiles(mcs_with_r)
        generated_smiles.add(smiles)
    print(f'1.{generated_smiles}')
           
    # Start processing the molecules (using i starting from 1)
    for i, mcs_with_r in enumerate(valid_molecules):
        smiles = Chem.MolToSmiles(mcs_with_r)

        if mcs_with_r is None:
            raise ValueError(f"Invalid input molecule (mcs_with_r is None), molecule index: {i + 1}")  # i starts from 1
        if is_complex_bicyclic(mcs_with_r):
            # print(f"Created bridged ring after expansion, skipping: {smile}")
            continue 
        # Create a copy of the molecule to avoid modifying the original object
        mol = Chem.RWMol(mcs_with_r)
        if mol is None:
            raise ValueError(f"Unable to create a molecule copy, molecule index: {i + 1}")  # i starts from 1

        # Get all asterisk atom indices
        asterisk_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        if not asterisk_indices:
            raise ValueError(f"No asterisk atoms in molecule {i + 1}")  # i starts from 1

        # Remove the asterisk atom at index 0 (if it exists)
        move_asterisk_idx = asterisk_indices[0]
        if move_asterisk_idx == 0:
            # Remove the original asterisk connections
            for bond in mol.GetAtomWithIdx(move_asterisk_idx).GetBonds():
                mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            mol.RemoveAtom(move_asterisk_idx)  # Remove the atom itself

        # Get connected atom information for the asterisk atom (after removal if index 0)
        connected_atoms = [bond.GetOtherAtomIdx(move_asterisk_idx)
                           for bond in mol.GetAtomWithIdx(move_asterisk_idx).GetBonds()]

        # Find valid positions for relocation
        valid_positions = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetIdx() != move_asterisk_idx:
                neighbors = [bond.GetOtherAtomIdx(atom.GetIdx()) for bond in atom.GetBonds()]

                # Exclude * and R atoms in the neighbors
                if all(mol.GetAtomWithIdx(n).GetSymbol() not in ['*', 'R'] for n in neighbors):
                    # Check if the atom can form an additional bond (i.e., has an available valence)
                    if atom.GetNumImplicitHs() > 0 or len(neighbors) < atom.GetTotalValence():
                        valid_positions.append(atom.GetIdx())
                    else:
                        # If the atom already has a connection to * (not index 0), we need to flag it
                        for bond in atom.GetBonds():
                            if bond.GetOtherAtomIdx(atom.GetIdx()) in asterisk_indices:
                                break

        if not valid_positions:
            continue

        try:
            for idx, pos in enumerate(valid_positions):
                # Create a deep copy of the original molecule to modify
                mol_copy = deepcopy(mol)  
                if is_complex_bicyclic(mol_copy):
                    # print(f"Created bridged ring after expansion, skipping: {smile}")
                    continue 

                # Add the new asterisk atom at a valid position
                new_asterisk_idx = mol_copy.AddAtom(Chem.Atom(0))  # Create new '*' atom
                mol_copy.AddBond(new_asterisk_idx, pos, Chem.BondType.SINGLE)  # Connect it to the selected position

                # Get SMILES for the modified molecule
                extend_smiles = Chem.MolToSmiles(mol_copy)

                if extend_smiles in generated_smiles:
                    continue
                if is_equivalent_to_existing(extend_smiles, generated_smiles):
                    continue
                generated_smiles.add(extend_smiles)
                
                # Sanitize the molecule to ensure it is valid
                Chem.SanitizeMol(mol_copy)

                # Compute 2D coordinates for visualization
                AllChem.Compute2DCoords(mol_copy)

                # Create an image of the molecule using your provided method
                drawer = Draw.MolDraw2DCairo(300, 300)
                options = drawer.drawOptions()
                options.useBWAtomPalette()  # Use black and white atom palette
                options.colorAtoms = False  # Do not color atoms
                options.highlightColor = None  # No highlight color

                drawer.DrawMolecule(mol_copy)
                drawer.FinishDrawing()

                # Save the image to the specified output folder
                png_data = drawer.GetDrawingText()
                img = Image.open(io.BytesIO(png_data)).convert("RGB")
                img_path = os.path.join(output_folder, f'mcs_with_r_{i + 1}_extended_{idx}_{pos}.png')  # i starts from 1
                img.save(img_path)

        except Exception as e:
            # Suppress the error and don't raise it or print anything
            pass  # Ignore any errors silently
        
    print(f'2.{generated_smiles}')
    
    return generated_smiles


def is_benzene_ring(mol, ring_atoms):
    """Determine if the ring is a benzene ring (aromatic six-membered carbon ring)"""
    if len(ring_atoms) != 6:
        return False
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or not atom.GetIsAromatic():
            return False
    bonds = mol.GetBonds()
    for bond in bonds:
        if bond.GetBeginAtomIdx() in ring_atoms and bond.GetEndAtomIdx() in ring_atoms:
            if bond.GetBondType() != Chem.BondType.AROMATIC:
                return False
    return True

def is_cyclohexane_ring(mol, ring_atoms):
    """Determine if the ring is a cyclohexane (non-aromatic six-membered single-bond carbon ring)"""
    if len(ring_atoms) != 6:
        return False
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return False
    bonds = mol.GetBonds()
    for bond in bonds:
        if bond.GetBeginAtomIdx() in ring_atoms and bond.GetEndAtomIdx() in ring_atoms:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return False
    return True

def adjust_nitrogen_hydrogens(atom):
    """Adjust the number of hydrogens on a nitrogen atom to match its valence"""
    if atom.GetSymbol() != 'N':
        return
    current_h = atom.GetNumExplicitHs()
    if current_h > 0:
        atom.SetNumExplicitHs(current_h - 1)
    else:
        implicit_h = atom.GetNumImplicitHs()
        if implicit_h > 0:
            atom.SetNumExplicitHs(implicit_h - 1)


def extend_mcs_with_ring(generated_smiles, output_folder):
    if isinstance(generated_smiles, set):
        generated_smiles = " ".join(generated_smiles)
    smiles_list = generated_smiles.split()
    
    for i, smile in enumerate(smiles_list):
        if smile.count('*') != 1:
            continue
        
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            continue
        
        if is_complex_bicyclic(mol):
            # print(f"Skipping complex bicyclic structure: {smile}")
            continue  
              
        try:
            Chem.Kekulize(mol)
        except Chem.KekulizeException:
            print("Kekulization failed, trying to continue")
        
        Chem.SanitizeMol(mol)
        
        ring_info = list(rdmolops.GetSymmSSSR(mol))
        reactive_atoms = set()
        
        # Step 1: Identify reactive atoms
        for ring in ring_info:
            ring_atoms = list(ring)
            if is_benzene_ring(mol, ring_atoms) or is_cyclohexane_ring(mol, ring_atoms):
                # print(f"Skipping excluded ring: {ring_atoms}")
                continue
            
            for atom_idx in ring_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                atom.UpdatePropertyCache()
                
                if atom.GetSymbol() not in ['C', 'N']:
                    continue
                
                used_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
                
                if atom.GetSymbol() == 'C':
                    remaining_valence = 4 - used_valence
                else:
                    explicit_degree = atom.GetDegree()
                    remaining_valence = 3 - explicit_degree
                    if remaining_valence < 1 and atom.GetNumImplicitHs() > 0:
                        remaining_valence += 1
                
                if remaining_valence >= 1:
                    reactive_atoms.add(atom_idx)
        
        # Step 2: Find adjacent pairs
        adj_pairs = []
        for atom_idx in reactive_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in reactive_atoms and neighbor_idx > atom_idx:
                    for ring in ring_info:
                        if atom_idx in ring and neighbor_idx in ring:
                            ring_size = len(ring)
                            adj_pairs.append( (atom_idx, neighbor_idx, ring_size) )
                            break
        
        if not adj_pairs:
            # print(f"No adjacent reactive pairs in {smile}")
            continue
        
        a1, a2, ring_size = adj_pairs[0]
        
        emol = Chem.EditableMol(mol)
        new_atoms = []
        
        if ring_size == 5:
            # Add a six-membered ring
            for _ in range(4):
                new_atoms.append(emol.AddAtom(Chem.Atom('C')))
            emol.AddBond(a1, new_atoms[0], Chem.BondType.SINGLE)
            emol.AddBond(new_atoms[0], new_atoms[1], Chem.BondType.SINGLE)
            emol.AddBond(new_atoms[1], new_atoms[2], Chem.BondType.SINGLE)
            emol.AddBond(new_atoms[2], new_atoms[3], Chem.BondType.SINGLE)
            emol.AddBond(new_atoms[3], a2, Chem.BondType.SINGLE)
        elif ring_size == 6:
            # Add a five-membered ring
            for _ in range(3):
                new_atoms.append(emol.AddAtom(Chem.Atom('C')))
            emol.AddBond(a1, new_atoms[0], Chem.BondType.SINGLE)
            emol.AddBond(new_atoms[0], new_atoms[1], Chem.BondType.SINGLE)
            emol.AddBond(new_atoms[1], new_atoms[2], Chem.BondType.SINGLE)
            emol.AddBond(new_atoms[2], a2, Chem.BondType.SINGLE)
        else:
            continue
        
        new_mol = emol.GetMol()
        
        # Check again after extension: did we create a complex fused/bridged ring?
        if is_complex_bicyclic(new_mol):
            # print(f"Created bridged ring after expansion, skipping: {smile}")
            continue   
             
        # Adjust hydrogens on nitrogen atoms
        for atom_idx in [a1, a2]:
            atom = new_mol.GetAtomWithIdx(atom_idx)
            adjust_nitrogen_hydrogens(atom)
        
        try:
            Chem.SanitizeMol(new_mol)
            AllChem.Compute2DCoords(new_mol)
        except Exception as e:
            # print(f"Sanitization error: {e}")
            continue

        # Create an image of the molecule using your provided method
        drawer = Draw.MolDraw2DCairo(300, 300)
        options = drawer.drawOptions()
        options.useBWAtomPalette()  # Use black and white atom palette
        options.colorAtoms = False  # Do not color atoms
        options.highlightColor = None  # No highlight color

        drawer.DrawMolecule(new_mol)
        drawer.FinishDrawing()

        # Save the image to the specified output folder
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data)).convert("RGB")
        img_path = os.path.join(output_folder, f'mcs_with_r_ring_{i + 1}_extended_{a1}_{a2}.png')  # i starts from 1 
        img.save(img_path)

    return


def is_complex_bicyclic(mol):
    """
    Check if the molecule contains overlapping rings (bridged or fused),
    suggesting it's a complex non-planar ring system.
    """
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    # Check for atom overlap across multiple rings (shared atoms)
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            shared = set(atom_rings[i]) & set(atom_rings[j])
            if len(shared) > 1:  # Shared more than 1 atom = fused or bridged ring
                return True
    return False

