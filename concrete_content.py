from collections import Counter, defaultdict
import io
import os
import json
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw, AllChem
from rdkit.Chem import rdmolops
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
logging.basicConfig(filename='mcs_finder.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def read_data(file_path):
    try:
        df = pd.read_csv(file_path, header=None) 
        logging.info(f"Successfully read {file_path}, number of rows: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None


def parse_smiles(df, output_folder):
    molecule_list = []
    molecule_index = []
    logging.info(f"Starting to parse SMILES from dataframe with {len(df)} rows")
    
    for i, cell in enumerate(df.iloc[:, 0]):
        logging.debug(f"Processing SMILES: {cell}") 
        try:
            molecule = Chem.MolFromSmiles(cell)
            if molecule:
                Chem.Kekulize(molecule, clearAromaticFlags=True)
                molecule_list.append(molecule)
                molecule_index.append((i, molecule))  
            else:
                logging.error(f"Unable to parse SMILES ({cell}): Failed to generate molecule")
        except Exception as e:
            logging.error(f"Unable to parse SMILES ({cell}): {e}")
    
    logging.info(f"Parsed {len(molecule_list)} molecules")
    
    for i, mol in molecule_index:
        img_path = os.path.join(output_folder, f'Molecule_{i}.png')
        
        drawer = Draw.MolDraw2DCairo(300, 300)
        options = drawer.drawOptions()
        options.useBWAtomPalette()       
        options.colorAtoms = False       
        options.bondLineWidth = 1       
        options.highlightColor = None    
        
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        with open(img_path, "wb") as f:
            f.write(drawer.GetDrawingText())
        
        logging.info(f"Saved structure image of Molecule {i} to {img_path}")

    return molecule_list, molecule_index


def find_mcs(molecule_list):
    if len(molecule_list) < 2:
        return None
    res = rdFMCS.FindMCS(molecule_list,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                         completeRingsOnly=True) 
    return res.smartsString if res else None


global_r_group_counter = 1
global_x_group_counter = 1
global_z_group_counter = 1


def mark_mcs_with_r(molecule_list, mcs_smiles, output_folder, differences):
    global global_r_group_counter
    global global_x_group_counter
    global global_z_group_counter

    mcs_mol = Chem.MolFromSmarts(mcs_smiles)
    if not mcs_mol:
        print("Error: MCS molecule creation failed from SMARTS.")
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
                    r_group_mapping[r_atom_idx] = f'R{global_r_group_counter}'
                    global_r_group_counter += 1
                r_group_atom = mcs_with_r.GetAtomWithIdx(r_atom_idx)
                r_group_atom.SetProp('atomLabel', r_group_mapping[r_atom_idx])
                for neighbor in r_group_atom.GetNeighbors():
                    atom_indices.add(neighbor.GetIdx())
            else:
                r_group_atom = Chem.Atom(0)
                r_group_idx = mcs_with_r.AddAtom(r_group_atom)
                mcs_with_r.AddBond(r_atom_idx, r_group_idx, Chem.BondType.SINGLE)
                r_group_mapping[r_group_idx] = f'R{global_r_group_counter}'
                mcs_with_r.GetAtomWithIdx(r_group_idx).SetProp('atomLabel', r_group_mapping[r_group_idx])
                atom_indices.add(r_atom_idx)
                global_r_group_counter += 1

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
                        x_group_mapping[atom_idx] = f'X{global_x_group_counter}'
                        global_x_group_counter += 1
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
                        z_group_mapping[atom_idx] = f'Z{global_z_group_counter}'
                        global_z_group_counter += 1
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
            print(f"Skipping empty X group {x_group}")
        else:
            create_x_group_folder(output_folder, x_group, replaced_atoms)
            x_groups_created.add(x_group)

    z_groups_created = set()  # Used to track created Z group folders
    for z_group, replaced_atoms in z_group_replacements.items():
        if not replaced_atoms:
            del z_group_mapping[z_group]
            print(f"Skipping empty Z group {z_group}")
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
        else:
            print("Warning: No substructure match found for molecule.")
        original_bond_mappings.append(bond_mapping)
    return original_bond_mappings


def shared_ring_bonds_cleavage(mol, atom_idx, connected_bonds, bond_map, original_atom_mappings, i, extra_indices_map):  
    bonded_bonds = []
    current_bond_idx = None
    bond_to_fragment = []
    bond_to_break = []

    c_key = None  
    d_key = None  

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if begin_idx in original_atom_mappings[i] and end_idx in original_atom_mappings[i]:
            if begin_idx == atom_idx or end_idx == atom_idx:
                other_atom_idx = end_idx if begin_idx == atom_idx else begin_idx
                bonded_bonds.append((bond.GetIdx(), other_atom_idx))
                current_bond_idx = bond.GetIdx()  

                extra_indices_a = extra_indices_map.get(atom_idx, [])
                extra_indices_b = extra_indices_map.get(other_atom_idx, [])                
                
                connected_bonds_a = [b.GetIdx() for b in mol.GetAtomWithIdx(atom_idx).GetBonds()]
                connected_bonds_b = [b.GetIdx() for b in mol.GetAtomWithIdx(other_atom_idx).GetBonds()]

                bond_rings = mol.GetRingInfo().BondRings()

                for c in extra_indices_a:   
                    for d in extra_indices_b:
                        common_rings = [ring for ring in bond_rings if c in ring and d in ring]
                        if common_rings:
                            c_key = c  
                            d_key = d  
                            bond_to_fragment_c = [b for b in connected_bonds_a if b != c_key]
                            bond_to_fragment_d = [b for b in connected_bonds_b if b != d_key]
                            
                            bond_to_fragment = bond_to_fragment_c + bond_to_fragment_d
                            bond_to_fragment = [b for b in bond_to_fragment if b != current_bond_idx]
                            bond_to_break.extend(bond_to_fragment)

    return bond_to_fragment, bond_to_break, c_key, d_key


def bonds_cleavage(mol, atom_idx, connected_bonds, bond_map, original_atom_mappings, i, c_key, d_key):
    # Get extra_indices, exclude keys in bond_map and exclude c_key and d_key
    extra_indices = [b for b in connected_bonds if b not in [bi[2] for bi in bond_map]]   
    extra_indices = [b for b in extra_indices if b not in {c_key, d_key}]

    # Get bonds in the rings
    ring_bonds = [b for ring in mol.GetRingInfo().BondRings() for b in ring]
    # Check if all extra_indices are in the same ring
    in_same_ring = all(b in ring_bonds for b in extra_indices)
    atom = mol.GetAtomWithIdx(atom_idx)
    atom.SetProp("molAtomMapNumber", str(atom_idx))

    # If they are in the same ring and the length of extra_indices is greater than 1
    if in_same_ring and len(extra_indices) > 1:
        bond_to_fragment = [b for b in connected_bonds if b not in extra_indices]
        bonds_to_break = []
        
        for bond_idx in bond_to_fragment:
            for other_bond in bond_to_fragment:
                bonds_to_break = set(bond_to_fragment)
                if other_bond != bond_idx:
                    if (mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx() in original_atom_mappings[i] and
                            mol.GetBondWithIdx(other_bond).GetBeginAtomIdx() in original_atom_mappings[i]) or \
                       (mol.GetBondWithIdx(bond_idx).GetEndAtomIdx() in original_atom_mappings[i] and
                            mol.GetBondWithIdx(other_bond).GetEndAtomIdx() in original_atom_mappings[i]):
                        bonds_to_break.add(other_bond)
        
        yield bond_to_fragment, list(bonds_to_break), extra_indices

    else:
        bond_to_fragment = extra_indices  

        if len(extra_indices) >= 2:
            for bond_idx in bond_to_fragment:
                bonds_to_break = [bond_idx] 
                yield [bond_idx], bonds_to_break, [bond_idx]               
        else:
            bonds_to_break = bond_to_fragment
            yield bond_to_fragment, bonds_to_break, extra_indices  


def handle_folder_creation(output_folder, r_groups):
    r_group_folder = "_".join(r_groups)
    atom_output_folder = os.path.join(output_folder, r_group_folder)
    
    # Rule 1: Single R-group case
    if len(r_groups) == 1:
        target_r = r_groups[0]
        # Find existing combined folders containing target R-group
        existing_combined = [
            f for f in os.listdir(output_folder)
            if os.path.isdir(os.path.join(output_folder, f)) and 
               target_r in f.split('_') and 
               len(f.split('_')) > 1
        ]
        if existing_combined:
            # Use first found combined folder
            return os.path.join(output_folder, existing_combined[0])

    # Rule 2: Multiple R-groups case
    else:
        # Create target folder first
        os.makedirs(atom_output_folder, exist_ok=True)
        
        # Merge individual R-group folders
        for r in r_groups:
            single_folder = os.path.join(output_folder, r)
            if os.path.exists(single_folder):
                # Move contents
                for item in os.listdir(single_folder):
                    src = os.path.join(single_folder, item)
                    dst = os.path.join(atom_output_folder, item)
                    if os.path.exists(dst):
                        os.remove(dst)  # Overwrite existing
                    shutil.move(src, dst)
                # Remove empty folder
                os.rmdir(single_folder)

    # Finalize folder path
    os.makedirs(atom_output_folder, exist_ok=True)
    return atom_output_folder


def fragment_and_draw(molecule_list, molecule_index, original_atom_mappings, 
                     original_bond_mappings, output_folder):
    """
    Generate molecular fragments based on original indices and draw images
    """
    r_group_to_atom_info = {}
    identical_fragments = defaultdict(dict)  # Using defaultdict for easy management

    for i, idx_info in enumerate(molecule_index):  
        original_idx = idx_info[0]  # Get the original molecule index
        mol = molecule_list[i]  # Get the corresponding molecule

        if i >= len(original_atom_mappings) or i >= len(original_bond_mappings):
            logging.error(f"Index mismatch: Molecule {original_idx} (position {i}) has no mapping data")
            continue

        atom_to_r_map = original_atom_mappings[i]  # Get the atom-to-R-group mapping
        bond_map = original_bond_mappings[i]  # Get the bond map
        extra_indices_map = {}

        # Process each atom and its R-group mapping
        for atom_idx, r_groups in atom_to_r_map.items():
            extra_indices_map[atom_idx] = [] 
            connected_bonds = []
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() == atom_idx or bond.GetEndAtomIdx() == atom_idx:
                    connected_bonds.append(bond.GetIdx())

            # Add connected bonds that are not in the bond map
            extra_indices_map[atom_idx] = [b for b in connected_bonds if b not in [bi[2] for bi in bond_map]]

        # Split the molecule and draw images for each atom and R-groups
        for atom_idx, r_groups in atom_to_r_map.items():
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_info = f"{atom.GetSymbol()} (Idx: {atom_idx})"
            
            # Record R-group information
            for r_group in r_groups:
                if r_group not in r_group_to_atom_info:
                    r_group_to_atom_info[r_group] = atom_info

            atom_output_folder = handle_folder_creation(output_folder, r_groups)
                                            

            connected_bonds = []
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() == atom_idx or bond.GetEndAtomIdx() == atom_idx:
                    connected_bonds.append(bond.GetIdx())
                    
            # Break shared bonds and handle ring-cleavage
            bond_to_fragment_shared, bonds_to_break_shared, c_key, d_key = shared_ring_bonds_cleavage(
                mol, atom_idx, connected_bonds, bond_map, original_atom_mappings, i, extra_indices_map)
            
            # Split molecule based on the shared bonds
            for bond_to_fragment_bonds, bonds_to_break_bonds, extra_indices in bonds_cleavage(
                mol, atom_idx, connected_bonds, bond_map, original_atom_mappings, i, c_key, d_key
            ):         
                if bond_to_fragment_shared:
                    # Fragment and draw image for shared bonds
                    images, frag_smiles, labels = fragment_and_draw_bond(
                        mol,
                        bond_to_fragment_shared, 
                        atom_idx,
                        r_groups,
                        i,
                        connected_bonds,
                        bonds_to_break_shared,  
                        bond_to_fragment_shared == extra_indices
                    )
                    if images:
                        combined_image = combine_images(images, labels)
                        # Generate fragment name based on molecule index and bond
                        bond_str = "_".join(map(str, sorted(bond_to_fragment_shared)))
                        base_name = f"Molecule_{original_idx}_Bond_{bond_str}_Fragments"
                        img_path = os.path.join(atom_output_folder, f"{base_name}.png")
                        combined_image.save(img_path)
                        with open(os.path.join(atom_output_folder, f"{base_name}.smiles"), 'w') as f:
                            f.write(frag_smiles[0])
                        logging.info(f"Saved combined fragment image to {img_path}")

                        # Check if bond-to-fragment is valid
                        if len(bond_to_fragment_shared) >= 2:  # Check the number of broken bonds
                            frag_key = str(tuple(sorted(bond_to_fragment_shared)))
                            if frag_key not in identical_fragments:
                                identical_fragments[frag_key] = {}
                            frag_info = identical_fragments[frag_key]
                            if frag_smiles[0] not in frag_info:
                                frag_info[frag_smiles[0]] = {
                                    "r_groups": r_groups,  # Store all R-groups
                                    "count": 1
                                }
                            else:
                                frag_info[frag_smiles[0]]["count"] += 1
                                frag_info[frag_smiles[0]]["r_groups"] = list(set(frag_info[frag_smiles[0]]["r_groups"] + r_groups))  # Merge R-groups

                    else:
                        logging.error(f"Failed to split Molecule {i+1} on Bond {bond_to_fragment_shared} into two fragments")

                # Handle individual bond fragmentation
                if bond_to_fragment_bonds:
                    for bond_idx in bond_to_fragment_bonds:
                        images, frag_smiles, labels = fragment_and_draw_bond(
                            mol,
                            [bond_idx], 
                            atom_idx,
                            r_groups,
                            i,
                            connected_bonds,
                            bonds_to_break_bonds,
                            bond_to_fragment_bonds == extra_indices
                        )
                        if images:
                            combined_image = combine_images(images, labels)
                            base_name = f"Molecule_{original_idx}_Bond_{bond_idx}_Fragments"
                            img_path = os.path.join(atom_output_folder, f"{base_name}.png")
                            combined_image.save(img_path)
                            with open(os.path.join(atom_output_folder, f"{base_name}.smiles"), 'w') as f:
                                f.write(frag_smiles[0])
                            logging.info(f"Saved combined fragment image to {img_path}")

                            # Record fragment information
                            if len(bonds_to_break_bonds) >= 2:  # Check the number of broken bonds
                                frag_key = str(bond_idx)
                                if frag_key not in identical_fragments:
                                    identical_fragments[frag_key] = {}
                                frag_info = identical_fragments[frag_key]
                                if frag_smiles[0] not in frag_info:
                                    frag_info[frag_smiles[0]] = {
                                        "r_groups": r_groups,  # Store all R-groups
                                        "count": 1,
                                        "same_atom": True
                                    }
                                else:
                                    frag_info[frag_smiles[0]]["count"] += 1
                                    frag_info[frag_smiles[0]]["r_groups"] = list(set(frag_info[frag_smiles[0]]["r_groups"] + r_groups))  # Merge R-groups

                        else:
                            logging.error(f"Failed to split Molecule {original_idx} on Bond {bond_idx} into two fragments")

    # Save global fragment information
    identical_fragments_file = os.path.join(output_folder, "identical_fragments.json")
    with open(identical_fragments_file, 'w') as f:
        json.dump(identical_fragments, f, indent=4)

    return r_group_to_atom_info


def fragment_and_draw_bond(mol, bond_idx, atom_idx, r_groups, molecule_idx, connected_bonds, bond_to_fragment, is_extra_indices):
    images = []
    frag_smiles = []
    labels = []
    successful_split = False
    
    try:
        mol.GetAtomWithIdx(atom_idx).SetProp("molAtomMapNumber", str(atom_idx))
        frag_mol = Chem.FragmentOnBonds(mol, bond_to_fragment)
        frag_mols = Chem.GetMolFrags(frag_mol, asMols=True)

        if len(frag_mols) >= 2:
            frag_to_keep = None
            if is_extra_indices:
                for frag in frag_mols:
                    if not any(atom.HasProp('molAtomMapNumber') and int(atom.GetProp('molAtomMapNumber')) == atom_idx for atom in frag.GetAtoms()):
                        frag_to_keep = frag
                        break
            else:
                for frag in frag_mols:
                    if any(atom.HasProp('molAtomMapNumber') and int(atom.GetProp('molAtomMapNumber')) == atom_idx for atom in frag.GetAtoms()):
                        frag_to_keep = frag
                        break

            if frag_to_keep is None:
                raise ValueError("No suitable fragment found based on the bond type.")
            for atom in frag_to_keep.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')

            drawer = Draw.MolDraw2DCairo(300, 300)
            options = drawer.drawOptions()
            options.useBWAtomPalette()    
            options.colorAtoms = False    
            options.highlightColor = None 

            img = Draw.MolToImage(frag_to_keep, size=(300, 300), kekulize=True, options=options)
            img = img.convert("RGB") 

            bg = Image.new("RGB", img.size, (255, 255, 255))
            diff = ImageChops.difference(img, bg)
            bbox = diff.getbbox()

            if bbox:
                img_cropped = img.crop(bbox)
            else:
                img_cropped = img 
            
            images.append(img_cropped)  
            frag_smiles.append(Chem.MolToSmiles(frag_to_keep))
            labels.append(f'Molecule {molecule_idx+1}, {r_groups}')
            successful_split = True

        if not successful_split:
            logging.error(f"Could not split Molecule {molecule_idx+1} into two fragments by breaking bonds connected to Atom {atom_idx}")

    except Exception as e:
        logging.error(f"Failed to fragment molecule {molecule_idx+1} on bond {bond_idx}: {e}")

    return images, frag_smiles, labels


def combine_images(images, labels):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights) + 50

    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    draw = ImageDraw.Draw(new_im)
    font = ImageFont.load_default()

    for i, im in enumerate(images):
        new_im.paste(im, (x_offset, 0))
        draw.text((x_offset, heights[i] + 10), labels[i], fill=(0, 0, 0), font=font)
        x_offset += im.width

    return new_im


def find_and_output_missing_fragments(output_folder, total_molecules):

    # Get all directories
    r_group_folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
    h_fragments = []

    for r_group_folder in r_group_folders:
        r_group_path = os.path.join(output_folder, r_group_folder)

        # Skip folders that don't contain "R"
        if "R" not in r_group_folder:
            continue

        # Determine the number of R groups based on the folder name
        r_groups = r_group_folder.split("_")
        required_fragments_per_molecule = len(r_groups)  # The number of fragments each Molecule_x should have

        # Iterate through each Molecule_x to check the number of fragments
        for i in range(1, total_molecules + 1):
            # Get existing fragment files
            fragment_files = [
                f for f in os.listdir(r_group_path) 
                if f"Molecule_{i}_" in f and f.endswith(".png")
            ]

            current_fragment_count = len(fragment_files)
            missing_fragments_count = required_fragments_per_molecule - current_fragment_count

            if missing_fragments_count > 0:
                logging.warning(
                    f"Molecule_{i} in {r_group_folder} is missing {missing_fragments_count} fragments."
                )
                for _ in range(missing_fragments_count):
                    h_atom_img = create_h_atom_image()
                    h_atom_img_path = os.path.join(
                        r_group_path, f"Molecule_{i}_Fragment_H_{current_fragment_count + 1}.png"
                    )
                    h_atom_img.save(h_atom_img_path)
                    logging.info(f"Saved H atom image: {h_atom_img_path}")

                    # Save SMILES file
                    h_smiles_path = os.path.join(
                        r_group_path, f"Molecule_{i}_Fragment_H_{current_fragment_count + 1}.smiles"
                    )
                    with open(h_smiles_path, "w") as f:
                        f.write("[H]")
                    logging.info(f"Saved H atom SMILES: {h_smiles_path}")

                    # Record the supplement information
                    h_fragments.append(
                        (f"Molecule_{i}_Fragment_H_{current_fragment_count + 1}.png", "[H]", r_group_path)
                    )

                    current_fragment_count += 1

    return h_fragments


def create_h_atom_image():
    h_atom = Chem.MolFromSmiles("[H]")
    img = Draw.MolToImage(h_atom, size=(300, 300), highlightColor=None, useBW=True)
    img = ImageOps.expand(img, border=50, fill='white')

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((150, 350), "H", fill=(0, 0, 0), font=font)
    
    return img


def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
        return standardized_smiles
    return None


def check_mcs(mcs_smiles, molecule_list):
    mcs_mol = Chem.MolFromSmarts(mcs_smiles)
    if not mcs_mol:
        print("Invalid MCS SMARTS.")
        return mcs_smiles, {}

    heavy_atom_count = sum(1 for atom in mcs_mol.GetAtoms() if atom.GetSymbol() not in ['H', 'D', 'T'])

    if heavy_atom_count <= 3:
        print("The number of heavy atoms is less than or equal to 3. Using the new method to retrieve MCS.")
        mcs_smiles, differences = find_mcs_2(molecule_list)  
        if mcs_smiles:
            mcs_smiles = mcs_smiles.replace("&!@", "").replace("&@", "")
    else:
        differences = {}

    return mcs_smiles, differences

def find_mcs_2(molecule_list):
    if len(molecule_list) < 2:
        return None, {}

    res = rdFMCS.FindMCS(
        molecule_list,
        atomCompare=rdFMCS.AtomCompare.CompareAny,  
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        completeRingsOnly=True
    )

    if res and res.smartsString:
        mcs_smarts = res.smartsString
        differences = record_atom_differences(molecule_list, mcs_smarts)
        return mcs_smarts, differences
    
    return None, {}

def record_atom_differences(molecule_list, mcs_smarts):
    differences = {}
    mcs_mol = Chem.MolFromSmarts(mcs_smarts)

    mcs_atom_indices = [atom.GetIdx() for atom in mcs_mol.GetAtoms()]

    for mol in molecule_list:
        match = mol.GetSubstructMatch(mcs_mol)
        if match:
            for mcs_idx, mol_idx in enumerate(match):
                atom_info = mol.GetAtomWithIdx(mol_idx).GetSymbol()
                if mcs_idx not in differences:
                    differences[mcs_idx] = set()
                differences[mcs_idx].add(atom_info)

    # Filter out atomic information with only one element
    differences = {k: v for k, v in differences.items() if len(v) > 1}

    return differences


def generate_fragments(file_path, output_folder):
    formula_folder = os.path.join(output_folder, 'formula')
    os.makedirs(formula_folder, exist_ok=True)
    
    # Process only a single file
    file_name = os.path.basename(file_path)
    logging.info(f'Processing file: {file_name}')

    all_fragments = []
    total_molecules = 0
    r_group_to_atom_info = {}

    # Read CSV file
    df = read_data(file_path)
    if df is None:
        logging.warning(f"Failed to read data from file: {file_name}")
        return total_molecules, r_group_to_atom_info

    # Parse SMILES strings and generate molecule list
    molecule_list, molecule_index = parse_smiles(df, output_folder)
    
    if len(molecule_list) < 2:
        logging.warning(f"Not enough molecules for MCS in file: {file_name}")
        return total_molecules, r_group_to_atom_info

    total_molecules = len(molecule_list)

    # Find Maximum Common Substructure (MCS)
    mcs_smiles = find_mcs(molecule_list)
    mcs_smiles = mcs_smiles.replace("&!@", "").replace("&@", "")
    if not mcs_smiles or '?' in mcs_smiles:
        logging.warning(f"Could not find a valid MCS: {file_name}")
        return total_molecules, r_group_to_atom_info
    mcs_smiles, differences = check_mcs(mcs_smiles, molecule_list)

    mcs_with_r, atom_indices, bond_indices, original_atom_mappings, original_bond_mappings = mark_mcs_with_r(
        molecule_list, mcs_smiles, output_folder, differences)
    
    if mcs_with_r:
        # Generate a larger image for cropping
        large_size = (800, 800)
        drawer = Draw.MolDraw2DCairo(large_size[0], large_size[1])
        options = drawer.drawOptions()
        options.useBWAtomPalette()    
        options.colorAtoms = False    
        options.highlightColor = None 
        drawer.DrawMolecule(mcs_with_r)
        drawer.FinishDrawing()

        png_data = drawer.GetDrawingText()
        mcs_img = Image.open(io.BytesIO(png_data))
        mcs_img = mcs_img.convert("RGB") 

        bg = Image.new("RGB", mcs_img.size, (255, 255, 255))
        diff = ImageChops.difference(mcs_img, bg)
        bbox = diff.getbbox()

        # Crop the image based on the content's bounding box
        if bbox:
            mcs_img_cropped = mcs_img.crop(bbox)
        else:
            mcs_img_cropped = mcs_img  # If no bounding box found, keep the original image

        # Save the cropped image
        mcs_img_path = os.path.join(formula_folder, f'MCS_{file_name}.png')
        mcs_img_cropped.save(mcs_img_path)
        logging.info(f'Saved marked MCS image to {mcs_img_path}')

        # Generate molecule fragments and images
        r_group_to_atom_info = fragment_and_draw(molecule_list, molecule_index, original_atom_mappings, original_bond_mappings, output_folder)
    else:
        logging.error(f"Failed to convert MCS to molecule object: {mcs_smiles}")

    if total_molecules > 0:
        h_fragments = find_and_output_missing_fragments(output_folder, total_molecules)
        all_fragments.extend(h_fragments)

    return total_molecules, r_group_to_atom_info


def new_file(output_folder, folder_name):
    """
    Create a new R folder at the specified path.
    """
    new_folder_path = os.path.join(output_folder, folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    logging.info(f"Created new folder: {new_folder_path}")
    return new_folder_path


def same_fragments(folder_path):
    logging.info(f"Checking similarity of fragments in folder: {folder_path}")
    molecule_fragments = defaultdict(list)

    for file in os.listdir(folder_path):
        if file.endswith('.smiles'):
            try:
                molecule_id = file.split('_')[1]
                with open(os.path.join(folder_path, file), 'r') as f:
                    smiles = f.read().strip() 
                    molecule_fragments[molecule_id].append(smiles)
            except IndexError:
                logging.warning(f"Skipping improperly named file: {file}")

    for molecule_id, fragments in molecule_fragments.items():
        # If the fragment contents are not identical, further processing is required
        if len(set(fragments)) > 1:
            logging.debug(f"Molecule_{molecule_id} has non-matching fragments: {set(fragments)}")
            return True

    logging.info("All fragments are identical. No further processing needed.")
    return False


def extract_features_from_smiles(smiles_list):
    features = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                features.append(np.array(fp))
            else:
                logging.error(f"Invalid molecule object: {smiles}")
        except Exception as e:
            logging.error(f"Unable to extract features: {e}")
    return np.array(features)


def read_smiles_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Unable to read file {file_path}: {e}")
        return None


def classify_molecule_1_fragments(folder_path, new_folder_paths):
    """
    Distribute Molecule_1 fragments into different new folders, ensuring each folder contains only one Molecule_1 fragment.
    """
    logging.info(f"Starting classification of Molecule_1 fragments, original folder: {folder_path}, target folders: {new_folder_paths}")

    r_group_folders = new_folder_paths  # Using the new folder paths passed in
    molecule_1_fragments = []

    # Find all Molecule_1 fragments
    for item in os.listdir(folder_path):
        if 'Molecule_1' in item and item.endswith('.smiles'):
            fragment_path = os.path.join(folder_path, item)
            molecule_1_fragments.append(fragment_path)

    if len(molecule_1_fragments) < len(r_group_folders):
        logging.error("Not enough Molecule_1 fragments to distribute into each folder")
        return

    # If there are multiple Molecule_1 fragments, ensure they are in different folders
    for i, fragment_file in enumerate(molecule_1_fragments):
        dest_folder = r_group_folders[i % len(r_group_folders)]  # Ensure distribution into different folders
        dest_smiles_path = os.path.join(dest_folder, os.path.basename(fragment_file))

        # Move the fragment
        shutil.move(fragment_file, dest_smiles_path)
        logging.info(f"Moved fragment: {fragment_file} to target folder: {dest_folder}")

        # Move corresponding PNG file
        img_file = fragment_file.replace('.smiles', '.png')
        if os.path.exists(img_file):
            dest_img_path = os.path.join(dest_folder, os.path.basename(img_file))
            shutil.move(img_file, dest_img_path)
            logging.info(f"Moved corresponding PNG file: {img_file} to target folder: {dest_folder}")

    logging.info("Molecule_1 fragment classification completed.")


def classify_fragments(folder_path, new_folder_paths):
    logging.info(f"Starting classification of Molecule_2 and subsequent fragments, original folder: {folder_path}, target folders: {new_folder_paths}")

    r_group_folders = new_folder_paths  # Using the new folder paths passed in

    # Extract features of Molecule_1 fragments for classification (without moving Molecule_1 fragments)
    classified_features = {}
    for r_folder in r_group_folders:
        r_folder_path = os.path.join(folder_path, r_folder)
        r_fragments = [os.path.join(r_folder_path, f) for f in os.listdir(r_folder_path)
                       if 'Molecule_1' in f and f.endswith('.smiles')]
        r_smiles = [read_smiles_from_file(f) for f in r_fragments]
        r_smiles = [smiles for smiles in r_smiles if smiles is not None]
        classified_features[r_folder] = extract_features_from_smiles(r_smiles)

    # Get all Molecule indexes (starting from 2)
    molecule_indexes = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.smiles') and filename.startswith('Molecule_'):
            parts = filename.split('_')
            try:
                index = int(parts[1].split('.')[0])
                molecule_indexes.add(index)
            except ValueError:
                pass

    # Classify fragments of each Molecule_x incrementally
    for mol_index in sorted(molecule_indexes):
        if mol_index == 1:
            continue  # Molecule_1 does not need classification

        mol_fragments = [os.path.join(folder_path, f)
                         for f in os.listdir(folder_path)
                         if f.startswith(f'Molecule_{mol_index}') and f.endswith('.smiles')]

        if not mol_fragments:
            logging.warning(f"Molecule_{mol_index} fragments are empty, skipping")
            continue

        mol_smiles = [read_smiles_from_file(f) for f in mol_fragments]
        mol_smiles = [smiles for smiles in mol_smiles if smiles is not None]
        
        # Check if all fragments of this Molecule_n are exactly the same
        unique_smiles = set(mol_smiles)
        
        if len(unique_smiles) == 1:
            # If all fragments are the same, directly assign them to different folders
            available_folders = list(r_group_folders)  # All folders
            random.shuffle(available_folders)  # Shuffle folder order

            for mol_idx, fragment in enumerate(mol_fragments):
                dest_folder = available_folders[mol_idx % len(available_folders)]  # Ensure distribution into different folders
                dest_folder_path = os.path.join(folder_path, dest_folder)
                os.makedirs(dest_folder_path, exist_ok=True)

                dest_smiles_path = os.path.join(dest_folder_path, os.path.basename(fragment))
                shutil.move(fragment, dest_smiles_path)
                logging.info(f"Moved fragment: {fragment} to target folder: {dest_folder}")

                img_file = fragment.replace('.smiles', '.png')
                if os.path.exists(img_file):
                    dest_img_path = os.path.join(dest_folder_path, os.path.basename(img_file))
                    shutil.move(img_file, dest_img_path)
                    logging.info(f"Moved corresponding PNG file: {img_file} to target folder: {dest_folder}")
            
            continue  # Skip similarity computation, directly classify into folders
        
        mol_features = extract_features_from_smiles(mol_smiles)

        used_folders = set()  # Track used folders

        # The key here is to ensure fragments with the same similarity are distributed into different folders
        for mol_idx, mol_feature in enumerate(mol_features):
            best_similarity = -1
            best_folder = None

            # Get all previously classified Molecule fragments, including Molecule_1 to Molecule_{mol_index-1}
            for r_folder, r_features in classified_features.items():
                if r_features.size == 0:
                    continue
                similarities = cosine_similarity(mol_feature.reshape(1, -1), r_features)
                max_similarity = np.max(similarities)

                # If a higher similarity is found, update the best folder
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_folder = r_folder
                elif max_similarity == best_similarity:
                    # If the similarity is the same, don't choose one folder, but distribute the fragments into different folders
                    available_folders = list(set(r_group_folders) - used_folders)
                    if available_folders:
                        best_folder = random.choice(available_folders)  # Randomly choose an unused folder
                    else:
                        best_folder = None  # If no folder is available, set it to None

            if best_folder is None:
                # If no best folder is found, choose an unused folder
                available_folders = list(set(r_group_folders) - used_folders)
                if available_folders:
                    best_folder = available_folders[0]  # Choose the first unused folder
                else:
                    logging.error(f"Cannot find a suitable folder for Molecule_{mol_index} fragment, skipping")
                    continue

            used_folders.add(best_folder)

            # Move fragment and PNG file
            src_path = mol_fragments[mol_idx]
            dest_folder_path = os.path.join(folder_path, best_folder)
            os.makedirs(dest_folder_path, exist_ok=True)

            dest_smiles_path = os.path.join(dest_folder_path, os.path.basename(src_path))
            shutil.move(src_path, dest_smiles_path)
            logging.info(f"Moved fragment: {src_path} to target folder: {best_folder}")

            img_file = src_path.replace('.smiles', '.png')
            if os.path.exists(img_file):
                dest_img_path = os.path.join(dest_folder_path, os.path.basename(img_file))
                shutil.move(img_file, dest_img_path)
                logging.info(f"Moved corresponding PNG file: {img_file} to target folder: {best_folder}")

    logging.info("Molecule_2 and subsequent fragment classification completed.")

            
def verify_r_folders(output_folder, total_molecules, r_groups):
    for r_group in r_groups:
        r_group_path = os.path.join(output_folder, r_group)
        if os.path.isdir(r_group_path):
            fragment_files = [f for f in os.listdir(r_group_path) if f.endswith('.png')]
            if len(fragment_files) != total_molecules:
                logging.error(f"Verification failed for {r_group}. Expected {total_molecules} fragments, found {len(fragment_files)}.")
                raise ValueError(f"Verification failed for {r_group}. Expected {total_molecules} fragments, found {len(fragment_files)}.")


def remove_empty_folders(path):
    if not os.path.isdir(path):
        return
    for f in os.listdir(path):
        fullpath = os.path.join(path, f)
        if os.path.isdir(fullpath):
            remove_empty_folders(fullpath)

    if not os.listdir(path):
        os.rmdir(path)
        logging.info(f"Deleted empty folder: {path}")


def process_fragments(output_folder):
    logging.info("Starting to process fragments in the output folder.")

    # Step 1: Find all folders containing "_"
    folders_to_process = [
        f for f in os.listdir(output_folder)
        if os.path.isdir(os.path.join(output_folder, f)) and "_" in f and len(f.split('_')) > 1
    ]
    logging.debug(f"Detected folders with '_': {folders_to_process}")

    # Step 2: Iterate over and process these folders
    for folder in folders_to_process:
        folder_path = os.path.join(output_folder, folder)

        # Check if further processing is needed
        need_processing = same_fragments(folder_path)

        # Step 3: Process based on the same_fragments result
        if not need_processing:
            logging.info(f"All fragments for folder '{folder}' are identical. Proceeding with new_file processing.")
            # Create new subfolders
            sub_folders = folder.split('_')  # Split folder name
            new_folder_paths = []
            for sub_folder in sub_folders:
                new_folder_path = new_file(output_folder, sub_folder)  # Create each new folder
                new_folder_paths.append(new_folder_path)

            # Copy content to the new folders
            for item in os.listdir(folder_path):
                src_item = os.path.join(folder_path, item)
                for new_folder_path in new_folder_paths:
                    if os.path.isdir(src_item):
                        shutil.copytree(src_item, os.path.join(new_folder_path, item), dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_item, new_folder_path)
            # Remove content from the original folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove subfolders
                else:
                    os.remove(item_path)  # Remove files
            logging.info(f"Folder '{folder}' processed as all fragments are identical.")
        else:
            logging.info(f"Fragments for folder '{folder}' are not identical. Proceeding with new_file processing.")
            # Create new subfolders
            sub_folders = folder.split('_')  # Split folder name
            new_folder_paths = []
            for sub_folder in sub_folders:
                new_folder_path = new_file(output_folder, sub_folder)  # Create each new folder
                new_folder_paths.append(new_folder_path)

            # Classify into new folders
            classify_molecule_1_fragments(folder_path, new_folder_paths)  # Classify Molecule_1
            classify_fragments(folder_path, new_folder_paths)  # Classify the remaining fragments
            logging.info(f"Folder '{folder}' processed with classification due to non-identical fragments.")

    # Step 4: Clean up folders containing "_"
    logging.info("Cleaning up folders with '_' in the name...")
    for folder in folders_to_process:
        folder_path = os.path.join(output_folder, folder)
        # Remove all folders containing "_"
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"Removed folder: {folder}")

    # Step 5: Clean up empty folders
    logging.info("Removing any remaining empty folders...")
    remove_empty_folders(output_folder)

    # Step 6: Finished processing
    logging.info("Finished processing fragments and cleaned up empty folders.")

    
def clean_smiles(smiles):
    bond_marker = re.findall(r'\[\d*\*\]', smiles)
    
    if len(bond_marker) == 1:
        cleaned_smiles = re.sub(r'\[\d*\*\]', '*', smiles)
        return cleaned_smiles
    
    elif len(bond_marker) >= 2:
        cleaned_smiles = re.sub(r'\[\d*\*\]', '[*]', smiles)
        cleaned_smiles = re.sub(r'\(\[\*\]\)', '', cleaned_smiles)  
        cleaned_smiles = re.sub(r'\=\[\*\]', '', cleaned_smiles)
        cleaned_smiles = re.sub(r'\#\[\*\]', '', cleaned_smiles)
        cleaned_smiles = re.sub(r'\[\*\]\#', '', cleaned_smiles)
        cleaned_smiles = re.sub(r'\[\*\]\=', '', cleaned_smiles)
        cleaned_smiles = re.sub(r'\[\*\]', '', cleaned_smiles)
        cleaned_smiles = '*' + cleaned_smiles
        return cleaned_smiles

    return smiles


def clean_smiles_1(smiles):
    cleaned_smiles = re.sub(r'\[\d*\*]', '*', smiles)
    bond_marker_num = len(re.findall(r'\[\*\]', cleaned_smiles))
    return cleaned_smiles, bond_marker_num


def clean_smiles_2(smiles):
    if not isinstance(smiles, str):
        raise TypeError("Expected a string for SMILES, got: {}".format(type(smiles)))
    cleaned_smiles = re.sub(r'\[\d*\*]', '[*]', smiles)
    bond_marker_num = len(re.findall(r'\[\*\]', cleaned_smiles))
    
    return cleaned_smiles, bond_marker_num


from description import description_data 


replace_map = {
    '\\': '',   # Replace backslash
    '+': '',    # Replace plus sign
    '-': '',    # Replace minus sign
    '/': '',    # Replace slash
    '*': '',    # Replace asterisk
    '[': '',    # Replace left bracket
    ']': '',    # Replace right bracket
    '@': '', 
    '#': '≡',
    '(H)': 'H',
    'N(=O)O': 'NO2',
    'C(=O)OH': 'COOH' ,
    '(=O)(=O)(=O)':  'O3',
    '(=O)(=O)':  'O2',
    '(=O)':  'O',   
    'C(F)(F)F' :'CF3', 
    '(2H)(2H)2H':'D3',
    '(3H)(3H)3H':'T3',     
    '2H': 'D',
    '3H': 'T',
    '(CH3)(CH3)CH3': '(CH3)3',
    '(CH3)CH3': '(CH3)2',
    'CH2CH2CH2CH2CH2CH2CH2CH2': '(CH2)8',
    'CH2CH2CH2CH2CH2CH2CH2': '(CH2)7',
    'CH2CH2CH2CH2CH2CH2': '(CH2)6',
    'CH2CH2CH2CH2CH2': '(CH2)5',
    'CH2CH2CH2CH2': '(CH2)4',
    'CH2CH2CH2': '(CH2)3',
    'CH2CH2': '(CH2)2',
    'C≡N': 'CN', 
}


def rule_description(output_folder, r_group_to_atom_info):  
    processed_folders = []
    
    # Step 1: Read the identical_fragments file, treat as empty if it doesn't exist
    identical_fragments_file = os.path.join(output_folder, "identical_fragments.json")
    identical_fragments = {}
    if os.path.exists(identical_fragments_file):
        with open(identical_fragments_file, 'r') as f:
            identical_fragments = json.load(f)

    # Step 2: Create the output file
    output_file_path = os.path.join(output_folder, "rule_description.txt")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        
        for folder in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder)
            if os.path.isdir(folder_path) and not folder.endswith('_description') and folder.startswith('R'):
                fragment_identifications = []
                unmatched_smiles = {}

                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.smiles') and file_name.startswith('Molecule_'):
                        fragment_file_path = os.path.join(folder_path, file_name)

                        try:
                            with open(fragment_file_path, 'r', encoding='utf-8') as file:
                                smiles = file.read().strip()  # Original SMILES
                                r_group = folder
                                cleaned_smiles = standardize_smiles(clean_smiles(smiles))  # Cleaned SMILES

                                shared_r_group_info = None
                                for frag_key, frag_data in identical_fragments.items():
                                    if smiles in frag_data:  
                                        shared_r_group_info = frag_data[smiles]
                                        break

                                matched_description = None
                                for entry in description_data:  
                                    smiles_list = []
                                    for sm in entry.get("SMILE", "").split(";"):
                                        cleaned_smile = standardize_smiles(clean_smiles(sm))
                                        if cleaned_smile is not None:
                                            smiles_list.append(cleaned_smile)
                                        else:
                                            smiles_list.append(sm)  
                                    
                                    if cleaned_smiles in smiles_list:
                                        matched_description = entry.get("Description")
                                        break

                                # Ensure the output includes forms content
                                if shared_r_group_info: 
                                    other_r_groups = [r for r in shared_r_group_info["r_groups"] if r != r_group]

                                    if shared_r_group_info.get("same_atom", False):
                                        if matched_description:  
                                            fragment_identifications.append(
                                                f"{file_name}: forms {matched_description} with {', '.join(other_r_groups)}"
                                            )
                                        else:
                                            fragment_identifications.append(
                                                f"{file_name}: forms other similar ring with {', '.join(other_r_groups)}"
                                            )
                                            if file_name not in unmatched_smiles:
                                                unmatched_smiles[file_name] = []
                                            unmatched_smiles[file_name].append(smiles)
                                    else:
                                        if matched_description:
                                            fragment_identifications.append(
                                                f"{file_name}: forms {matched_description} with {', '.join(other_r_groups)}"
                                            )
                                        else:
                                            fragment_identifications.append(
                                                f"{file_name}: forms other similar ring with {', '.join(other_r_groups)}"
                                            )
                                            if file_name not in unmatched_smiles:
                                                unmatched_smiles[file_name] = []
                                            unmatched_smiles[file_name].append(smiles)
                                else:
                                    if matched_description:
                                        fragment_identifications.append(f"{file_name}: {matched_description}")
                                    else:
                                        if file_name not in unmatched_smiles:
                                            unmatched_smiles[file_name] = []
                                        unmatched_smiles[file_name].append(smiles)

                                        # Process simple structure fragments
                                        mol = Chem.MolFromSmiles(cleaned_smiles)
                                        if mol:
                                            ring_info = mol.GetRingInfo()
                                            rings = ring_info.AtomRings()

                                            if not rings:
                                                mol_with_h = Chem.AddHs(mol)  # Add hydrogen atoms
                                                mol_formula = Chem.MolToSmiles(mol_with_h, canonical=True)

                                                # Handle hydrogen representation in molecular formula
                                                mol_formula = re.sub(r'\[H\]', 'H', mol_formula)
                                                mol_formula = re.sub(r'\(H\)', 'H', mol_formula)
                                                mol_formula = re.sub(r'H+', lambda m: f'H{len(m.group(0))}' if len(m.group(0)) > 1 else 'H', mol_formula)

                                                for old_char, new_char in replace_map.items():
                                                    mol_formula = mol_formula.replace(old_char, new_char)

                                                if mol_formula.startswith('='):
                                                    fragment_identifications.append(f"{file_name}: {mol_formula}")
                                                else:
                                                    fragment_identifications.append(f"{file_name}: -{mol_formula}")
                                                    
                                                if file_name in unmatched_smiles and smiles in unmatched_smiles[file_name]:
                                                    unmatched_smiles[file_name].remove(smiles)
                                                    if not unmatched_smiles[file_name]:
                                                        del unmatched_smiles[file_name] 
                                            else:
                                                fragment_identifications.append(f"{file_name}: No matching description found")
                                        else:
                                            fragment_identifications.append(f"{file_name}: Invalid SMILES format")

                        except Exception as e:
                            error_message = f"Error reading file: {fragment_file_path}, {e}"
                            print(error_message)
                            fragment_identifications.append(f"{file_name}: Error processing fragment")

                if fragment_identifications:
                    processed_folders.append({
                        "folder": folder,
                        "file_names": unmatched_smiles,
                        "identifications": fragment_identifications
                    })

        # Sort processed folders in descending order by the first numeric part in the folder name
        processed_folders.sort(key=lambda x: int(x["folder"].split('_')[0][1:]), reverse=False)

        for entry in processed_folders:
            output_file.write(f"Folder: {entry['folder']}\n")
            for identification in entry["identifications"]:
                output_file.write(f"  {identification}\n")

    return processed_folders



def find_ring(mol, bond_marker_atoms):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    shared_bond_markers = {}
    bond_marker_connected_rings = set()

    for ring_idx, ring in enumerate(atom_rings):
        bond_marker_connected = False

        for atom_idx in ring:
            if atom_idx in bond_marker_atoms:
                bond_marker_connected = True

            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            for nei in neighbors:
                nei_idx = nei.GetIdx()
                if nei_idx in ring:
                    bond_pair = tuple(sorted([atom_idx, nei_idx]))

                    if bond_pair not in shared_bond_markers:
                        shared_bond_markers[bond_pair] = [ring_idx]
                    else:
                        if ring_idx not in shared_bond_markers[bond_pair]:
                            shared_bond_markers[bond_pair].append(ring_idx)

        if bond_marker_connected:
            bond_marker_connected_rings.add(ring_idx)

    return shared_bond_markers, bond_marker_connected_rings


def analyze_ring_info(mol, bond_marker_atoms):
    shared_bond_markers, bond_marker_connected_rings = find_ring(mol, bond_marker_atoms)

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    ring_data = []

    for ring_idx, ring in enumerate(atom_rings):
        ring_bond_types = []
        non_carbon_atoms = []  
        num_non_carbon_atoms = 0
        num_double_bonds = 0
        bond_marker_connected = ring_idx in bond_marker_connected_rings
        atom_order = []  # Record the atom order in the ring
        atomic_connections = []  # Record atomic connection information

        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_symbol = atom.GetSymbol()

            # Record the atom type order in the ring
            atom_order.append(atom_symbol)

            # Get the neighbors of the current atom
            neighbors = atom.GetNeighbors()
            atom_hydrogen_info = None  # Used to store hydrogen atom information of the current atom

            for nei in neighbors:
                nei_idx = nei.GetIdx()
                if nei_idx in ring:
                    bond = mol.GetBondBetweenAtoms(atom_idx, nei_idx)
                    bond_type = bond.GetBondType()
                    ring_bond_types.append(bond_type)

                    # Count the number of double bonds
                    if bond_type == Chem.rdchem.BondType.DOUBLE:
                        num_double_bonds += 1

                    # Record the atomic connection information, including aromatic bonds
                    atom1_symbol = atom.GetSymbol()
                    atom2_symbol = nei.GetSymbol()

                    # Check if it is an aromatic ring, aromatic ring bond type is AROMATIC
                    if bond_type == Chem.rdchem.BondType.AROMATIC:
                        connection_info = f"{atom1_symbol}-{atom2_symbol}-AROMATIC"
                    else:
                        connection_info = f"{atom1_symbol}-{atom2_symbol}-{bond_type.name}"

                    atomic_connections.append(connection_info)

            # Record hydrogen information for non-carbon atoms
            hydrogen_count = atom.GetTotalNumHs()
            if hydrogen_count > 0:
                if hydrogen_count == 1:
                    atom_hydrogen_info = f"{atom_symbol}-H"
                elif hydrogen_count == 2:
                    atom_hydrogen_info = f"{atom_symbol}-H2"
                else:
                    atom_hydrogen_info = f"{atom_symbol}-H{hydrogen_count}"

            # If it is a non-carbon atom, add it to non_carbon_atoms and record hydrogen information
            if atom_symbol != 'C':
                if atom_hydrogen_info:
                    non_carbon_atoms.append((ring.index(atom_idx), atom_symbol, atom_hydrogen_info))
                else:
                    non_carbon_atoms.append((ring.index(atom_idx), atom_symbol))
                num_non_carbon_atoms += 1

        # Update the ring information
        ring_info = {
            'ring_idx': ring_idx,
            'ring_size': len(ring),
            'bond_info': tuple(sorted(ring_bond_types)),
            'is_shared': False,  
            'non_carbon_atoms': tuple(sorted(non_carbon_atoms)),  # Record non-carbon atoms and their hydrogen information
            'num_non_carbon_atoms': num_non_carbon_atoms,
            'num_double_bonds': num_double_bonds,
            'bond_marker_connected': bond_marker_connected,
            'atom_order': tuple(atom_order),  # Record the atom order in the ring
            'atomic_connections': tuple(sorted(atomic_connections)),  # Record atomic connection methods and hydrogen atom information
            'ring_atoms': ring  
        }

        ring_data.append(ring_info)

    # Process the shared ring markers
    for bond_pair, shared_ring_indices in shared_bond_markers.items():
        if len(shared_ring_indices) > 1:  
            for ring_idx in shared_ring_indices:
                ring_data[ring_idx]['is_shared'] = True

    # Filter the ring data and process the shared rings
    filtered_ring_data = []
    rings_to_process = list(bond_marker_connected_rings)
    processed_rings = set()

    while rings_to_process:
        current_ring_idx = rings_to_process.pop()
        if current_ring_idx not in processed_rings:
            filtered_ring_data.append(ring_data[current_ring_idx])
            processed_rings.add(current_ring_idx)
            for bond_pair, shared_ring_indices in shared_bond_markers.items():
                if current_ring_idx in shared_ring_indices:
                    for idx in shared_ring_indices:
                        if idx != current_ring_idx and idx not in processed_rings:
                            rings_to_process.append(idx)

    return filtered_ring_data


def sanitize_filename(smiles):
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', smiles)
    return sanitized


def nest_fragments(output_folder, r_group_to_atom_info, total_molecules):
    processed_folders = rule_description(output_folder, r_group_to_atom_info)
    nested_fragments = []

    for folder_info in processed_folders:
        folder = folder_info['folder']
        folder_path = os.path.join(output_folder, folder)  # Get the path of the current folder
        file_names = folder_info['file_names']

        if not file_names:
            continue

        # Extract valid SMILES
        valid_entries = [
            {'file_name': file_name, 'unmatched_smiles': [s for s in smiles_list if Chem.MolFromSmiles(s)]}
            for file_name, smiles_list in file_names.items()
        ]
        valid_entries = [entry for entry in valid_entries if entry['unmatched_smiles']]

        if not valid_entries:
            print(f"No valid SMILES in {folder}, skipping")
            continue

        # Create nested folders and process
        nest_folder_path = os.path.join(folder_path, f"{folder}_nest")
        os.makedirs(nest_folder_path, exist_ok=True)
        fragments, mcs_mapping = process_unmatched_smiles(valid_entries, nest_folder_path)
        find_and_output_missing_fragments(nest_folder_path, total_molecules)
        process_fragments(nest_folder_path)

        new_processed_folders = rule_description(nest_folder_path, r_group_to_atom_info)

        if new_processed_folders:
            new_processed_folders, nest_folder_path = process_nested_folders(new_processed_folders, nest_folder_path, r_group_to_atom_info, total_molecules)

        # Save MCS mapping information
        mapping_file_path = os.path.join(nest_folder_path, 'mcs_mapping.json')
        with open(mapping_file_path, 'w', encoding='utf-8') as f:
            json.dump(mcs_mapping, f, indent=4)

        remove_empty_folders(nest_folder_path)
        nested_fragments.extend(fragments)

    return nested_fragments

        
def process_nested_folders(processed_folders, current_folder_path, r_group_to_atom_info, total_molecules):
    nest_processed_folders = []
    
    for folder_info in processed_folders:
        new_folder = folder_info['folder']
        nest_folder_path = os.path.join(current_folder_path, new_folder)
        new_folder_path = os.path.join(nest_folder_path, f"{new_folder}_nest")  # Create a new folder under the current nested folder path
        os.makedirs(new_folder_path, exist_ok=True)
        
        nest_file_names = folder_info['file_names']

        if not nest_file_names:
            print(f"All fragments in folder '{new_folder}' matched. Skipping further processing.")
            continue

        # Extract new valid SMILES
        nest_valid_entries = [
            {'file_name': file_name, 'unmatched_smiles': [s for s in smiles_list if Chem.MolFromSmiles(s)]}
            for file_name, smiles_list in nest_file_names.items()
        ]
        nest_valid_entries = [entry for entry in nest_valid_entries if entry['unmatched_smiles']]

        if not nest_valid_entries:
            print(f"No valid SMILES in {new_folder}, skipping")
            continue

        # Create new nested folders and process
        fragments, mcs_mapping = process_unmatched_smiles(nest_valid_entries, new_folder_path)
        find_and_output_missing_fragments(new_folder_path, total_molecules)
        process_fragments(new_folder_path)

        # Recursively process new nested folders to ensure processing in the parent folder path
        new_processed_folders = rule_description(new_folder_path, r_group_to_atom_info)
        if new_processed_folders:
            nested_folders, _ = process_nested_folders(new_processed_folders, new_folder_path, r_group_to_atom_info, total_molecules)
            nest_processed_folders.extend(nested_folders)
    
    return nest_processed_folders, new_folder_path


def remove_empty_folders(folder_path):
    """Recursively delete empty folders"""
    # Traverse all items in the folder
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        # If it's a folder, recursively call
        if os.path.isdir(entry_path):
            remove_empty_folders(entry_path)
    # If the folder is empty now, delete it
    if not os.listdir(folder_path):
        shutil.rmtree(folder_path)

   
   
def process_unmatched_smiles(unmatched_smiles_with_filenames, nest_folder_path):
    global global_r_group_counter
    fragments = []
    mcs_mapping = {}

    for entry in unmatched_smiles_with_filenames:
        file_name = entry['file_name']
        unmatched_smiles = entry['unmatched_smiles'] 

        if not unmatched_smiles:
            print(f"No unmatched SMILES found in {file_name}")
            continue
        
        for smiles in unmatched_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                logging.info(f"Successfully processed SMILES: {smiles}")
                
                bond_marker_positions = {}
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == '*':  
                        # Get the first adjacent atom connected to the marker atom
                        neighbors = atom.GetNeighbors()
                        if neighbors:
                            bond_marker_positions[atom.GetIdx()] = neighbors[0].GetIdx()

                for marker_idx, atom_idx in bond_marker_positions.items():
                    atom = mol.GetAtomWithIdx(atom_idx)
                    # Get bond information in the molecule
                    bond_info = next(({'bond_type': bond.GetBondTypeAsDouble(), 'bond_idx': bond.GetIdx()} 
                                      for bond in atom.GetBonds() 
                                      if bond.GetBeginAtomIdx() == marker_idx or bond.GetEndAtomIdx() == marker_idx), None)

                    # Store molecular information in fragment_info
                    fragment_info = {
                        'smiles': smiles,
                        'mol': mol,
                        'atom_info': {'atom_idx': atom_idx, 'atom_symbol': atom.GetSymbol(), 'bond_info': bond_info},
                        'ring_info': analyze_ring_info(mol, [atom_idx]) if atom.IsInRing() else None,
                        'file_name': file_name  
                    }
                    fragments.append(fragment_info)
            else:
                logging.warning(f"Failed to process SMILES: {smiles}")
    
    ring_fragments = [f for f in fragments if f['ring_info']]
    chain_fragments = [f for f in fragments if not f['ring_info']]
    
    if ring_fragments:
        ring_fragments, ring_mapping = nest_ring(ring_fragments, nest_folder_path)  
        mcs_mapping.update(ring_mapping)

    if chain_fragments:
        chain_fragments, chain_mapping = nest_chain(chain_fragments, nest_folder_path) 
        mcs_mapping.update(chain_mapping)
        
    mark_nest_mcs_with_r(mcs_mapping, nest_folder_path)

    return fragments, mcs_mapping     


def is_simple_mcs(mcs_mol):
    atom_count = mcs_mol.GetNumAtoms()
    heavy_atom_count = sum(1 for atom in mcs_mol.GetAtoms() if atom.GetAtomicNum() != 1)  
    if heavy_atom_count <= 2:
        return True
    return False


def generate_nest_image(mcs_mapping, folder_path, prefix="mcs_ring"):
    """
    Generate images for all MCS structures in mcs_mapping and save them in the specified folder path.
    prefix: Prefix to be used for the image names (default: 'mcs_ring').
    """
    mcs_counter = 1
    image_paths = []  # Store paths of all generated images

    # Iterate over the mcs_mapping to generate images
    for mcs_smiles, mapping_info in mcs_mapping.items():
        mcs_smiles = mapping_info['mcs_structure']
        group_fragments = mapping_info['fragments']
        
        mcs_mol = Chem.MolFromSmiles(mcs_smiles)

        drawer = Draw.MolDraw2DCairo(300, 300)
        options = drawer.drawOptions()
        options.useBWAtomPalette()
        options.colorAtoms = False
        options.highlightColor = None
        drawer.DrawMolecule(mcs_mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        img = img.convert("RGB")

        # Generate image path
        img_name = f"{prefix}_{mcs_counter}.png"
        img_path = os.path.join(folder_path, img_name)
        img.save(img_path)

        # Add the image path to the list
        image_paths.append(img_path)

        # Increment the counter for naming the next image
        mcs_counter += 1

    return image_paths


def nest_ring(fragments, folder_path):
    fragment_groups = {}
    mcs_counter = 1
    failed_fragments = []
    mcs_mapping = {}

    # Step 1: Group fragments by atom_info and ring_info
    for fragment in fragments:
        atom_info = fragment.get('atom_info', [])
        ring_info = fragment.get('ring_info', [])
        file_name = fragment.get('file_name', 'Unknown_File')

        atom_info_key = tuple(sorted([atom['symbol'] for atom in atom_info if isinstance(atom, dict) and 'symbol' in atom]))
        ring_info_key = tuple(
            sorted(
                (
                    info.get('ring_size', None),
                    tuple(info.get('bond_info', [])),
                    tuple(info.get('non_carbon_atoms', []))
                )
                for info in ring_info if isinstance(info, dict)
            )
        )

        group_key = (atom_info_key, ring_info_key)

        if group_key not in fragment_groups:
            fragment_groups[group_key] = []

        fragment_groups[group_key].append(fragment)

    # Step 2: Process each group of fragments
    for group_key, group_fragments in fragment_groups.items():
        if len(group_fragments) < 2:
            for frag in group_fragments:
                failed_fragments.append(frag)
            continue

        # Step 3: Prepare molecules and kekulize
        mol_list = [fragment['mol'] for fragment in group_fragments if fragment.get('mol') is not None]
        if len(mol_list) < 2:
            for frag in group_fragments:
                failed_fragments.append(frag)
            continue

        kekulized_mols = []
        for mol in mol_list:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                kekulized_mols.append(mol)
            except Exception:
                frag = group_fragments[mol_list.index(mol)]
                failed_fragments.append(frag)
                continue

        if len(kekulized_mols) < 2:
            for frag in group_fragments:
                failed_fragments.append(frag)
            continue

        # Step 4: Get bond_marker_atoms for MCS computation
        atom_info1 = group_fragments[0].get('atom_info', [])
        if isinstance(atom_info1, dict):
            bond_marker_atoms = [atom_info1.get('atom_idx')]
        elif isinstance(atom_info1, list):
            bond_marker_atoms = [atom.get('atom_idx') for atom in atom_info1 if isinstance(atom, dict)]
        else:
            continue

        mcs_params = rdFMCS.MCSParameters()
        mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
        mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrderExact
        mcs_params.CompleteRingsOnly = True
        mcs_params.AtomIdxLocks = bond_marker_atoms

        # Step 5: Compute MCS
        try:
            mcs_result = rdFMCS.FindMCS(kekulized_mols, parameters=mcs_params)
        except Exception:
            for frag in group_fragments:
                failed_fragments.append(frag)
            continue

        if mcs_result.canceled:
            for frag in group_fragments:
                failed_fragments.append(frag)
            continue

        # Step 6: Generate the MCS SMILES and save image
        mcs_smarts = mcs_result.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)

        if mcs_mol:
            mcs_smiles = Chem.MolToSmiles(mcs_mol)

            mcs_mapping[mcs_smiles] = {
                'mcs_structure': mcs_smiles,
                "fragments": []
            }

            # Collect the SMILES and file_name for each fragment in the group
            for frag in group_fragments:
                frag_smiles = frag.get('smiles', 'Unknown_SMILES')
                frag_file_name = frag.get('file_name', 'Unknown_File')

                mcs_mapping[mcs_smiles]["fragments"].append({
                    "smiles": frag_smiles,
                    "file_name": frag_file_name
                })
        else:
            failed_fragments.extend(group_fragments)
            
    # Step 7: Check and update MCS SMILES for missing '*' atoms
    for mcs_smiles, group_info in mcs_mapping.items():
        if "*" not in mcs_smiles:
            print(f"Processing ring MCS without '*' atom: {mcs_smiles}")
            mcs_mapping = generate_mcs_from_ring(group_info['fragments'], mcs_mapping)
            
    generate_nest_image(mcs_mapping, folder_path, prefix="mcs_ring")
                

    # Step 8: Handle failed fragments
    if failed_fragments:
        for frag in failed_fragments:
            smiles = frag.get('smiles', 'Unknown_SMILES')
            cleaned_smiles = clean_smiles_2(smiles)[0]
            sanitized_smiles = sanitize_filename(cleaned_smiles)
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol:
                # Identify ring atoms
                ring_info = mol.GetRingInfo()
                ring_atoms = set([atom_idx for ring in ring_info.AtomRings() for atom_idx in ring])

                # Remove non-ring atoms and bonds, but retain "*" markers
                atoms_to_remove = []
                for atom in mol.GetAtoms():
                    if atom.GetIdx() not in ring_atoms and atom.GetSymbol() != "*":
                        atoms_to_remove.append(atom.GetIdx())

                # Remove non-ring atoms and corresponding bonds
                mol = Chem.EditableMol(mol)
                for atom_idx in sorted(atoms_to_remove, reverse=True):
                    mol.RemoveAtom(atom_idx)
                mol = mol.GetMol()

                # Generate and save image with retained break markers
                drawer = Draw.MolDraw2DCairo(300, 300)
                options = drawer.drawOptions()
                options.useBWAtomPalette()    
                options.colorAtoms = False    
                options.highlightColor = None 
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()

                png_data = drawer.GetDrawingText()
                img = Image.open(io.BytesIO(png_data))
                img = img.convert("RGB")

                # 保存图像
                img_path = os.path.join(folder_path, f'failed_{sanitized_smiles}.png')
                img.save(img_path)

                # Add ring-only structure with "*" markers to mcs_mapping
                ring_only_smiles = Chem.MolToSmiles(mol)
                mcs_mapping[ring_only_smiles] = {
                    'mcs_structure': ring_only_smiles,
                    'fragments': [{
                        "smiles": frag['smiles'],
                        "file_name": frag['file_name']
                    }]
                }
    
    return fragments, mcs_mapping


def generate_mcs_from_ring(group_fragments, mcs_mapping):
    for mcs_smiles, group_info in mcs_mapping.items():

        # Store the extraction results for each fragment
        extracted_smiles_list = []

        # Get the SMILES for each molecular fragment
        for frag in group_info['fragments']:
            smiles = frag.get('smiles', '')
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                print(f"Error: Unable to generate molecule from SMILES: {smiles}")
                continue
            try:
                # Find all atoms marked as '*'
                star_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
                
                if not star_atoms:
                    print("Error: No '*' atoms found in the molecule.")
                    continue

                # Extract all ring structures
                ring_info = mol.GetRingInfo().AtomRings()
                connected_rings = set()
                for star_atom in star_atoms:
                    # Get neighbors of the '*' atom
                    star_neighbors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(star_atom).GetNeighbors()]
                    for neighbor in star_neighbors:
                        for ring in ring_info:
                            if neighbor in ring:
                                connected_rings.add(tuple(ring))
                                break

                if not connected_rings:
                    print("Error: No connected rings found.")
                    continue
                
                mol_copy = Chem.RWMol(mol)

                # Extract '*' atoms and the connected rings
                atoms_to_keep = set()
                for ring in connected_rings:
                    atoms_to_keep.update(ring)
                atoms_to_keep.update(star_atoms)

                # Remove atoms not in atoms_to_keep
                atoms_to_remove = [atom.GetIdx() for atom in mol_copy.GetAtoms() if atom.GetIdx() not in atoms_to_keep]
                for atom_idx in sorted(atoms_to_remove, reverse=True):
                    mol_copy.RemoveAtom(atom_idx)

                # Get the new SMILES
                new_mcs_smiles = Chem.MolToSmiles(mol_copy, isomericSmiles=True, kekuleSmiles=True)
                extracted_smiles_list.append(new_mcs_smiles)

            except Exception as e:
                print(f"Error processing molecule: {e}")
                continue

        # If the extracted results list is empty, skip this group
        if not extracted_smiles_list:
            print(f"Error: No valid MCS SMILES extracted for group {mcs_smiles}.")
            continue

        # Count the extracted results and select the most frequent SMILES
        smiles_counter = Counter(extracted_smiles_list)
        most_common_smiles = smiles_counter.most_common(1)[0][0]
        most_common_smiles = re.sub(r'\[\d+\*]', '*', most_common_smiles)

        # Update the mcs_mapping with the mcs_structure
        mcs_mapping[mcs_smiles]["mcs_structure"] = most_common_smiles

    return mcs_mapping


def nest_chain(fragments, folder_path):
    global global_r_group_counter 

    mcs_counter = 1
    mcs_mapping = {}
    handle_fragments = {}
    r_labels_set = set()  # Store unique R labels

    max_r_in_fragments = global_r_group_counter  

    for fragment in fragments:
        mol = fragment['mol']
        original_smiles = fragment['smiles']
        file_name = fragment.get('file_name', 'Unknown')  # Ensure file_name is available

        cleaned_smiles, bond_marker_num = clean_smiles_1(original_smiles)
        mol = Chem.MolFromSmiles(cleaned_smiles)
        bond_marker_idx = None

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                bond_marker_idx = atom.GetIdx()
                break

        if bond_marker_idx is None:
            print(f"No bond marker found in fragment: {cleaned_smiles}")
            continue

        visited_atoms = set()
        bonds_to_break = set()
        r_counter = global_r_group_counter

        def trace_mcs(atom_idx):
            atom = mol.GetAtomWithIdx(atom_idx)
            visited_atoms.add(atom_idx)
            for bond in atom.GetBonds():
                neighbor_atom = bond.GetOtherAtom(atom)
                neighbor_idx = neighbor_atom.GetIdx()
                if neighbor_idx not in visited_atoms:
                    ring_info = mol.GetRingInfo()
                    atom_rings = ring_info.AtomRings()
                    if any(neighbor_idx in ring for ring in atom_rings):
                        bonds_to_break.add(bond.GetIdx())
                        continue
                    trace_mcs(neighbor_idx)

        trace_mcs(bond_marker_idx)
        fragmented_mol = Chem.FragmentOnBonds(mol, list(bonds_to_break))
        fragments_with_rings = Chem.GetMolFrags(fragmented_mol, asMols=True)

        ring_fragments = []
        non_ring_fragments = []
        for frag in fragments_with_rings:
            if frag.GetRingInfo().AtomRings():
                ring_fragments.append(frag)
            else:
                non_ring_fragments.append(frag)

        if non_ring_fragments:
            for non_ring_frag in non_ring_fragments:
                non_ring_smiles = Chem.MolToSmiles(non_ring_frag, canonical=True)
                atom_replacement_map = {}

                extra_markers = re.findall(r'\[\d*\*]', non_ring_smiles)
                r_group_mapping = {}
                if extra_markers:
                    r_label = f'R{r_counter}'
                    r_labels_set.add(r_label)  # Add to unique set
                    for marker in extra_markers:
                        non_ring_smiles = non_ring_smiles.replace(marker, r_label)
                        r_group_mapping[str(r_counter)] = r_label

                possible_atoms = ['F', 'Cl', 'Br', 'I']
                existing_atoms = set(re.findall(r'[FClBrI]', non_ring_smiles))
                available_atoms = [atom for atom in possible_atoms if atom not in existing_atoms]

                def replace_r_with_atom(smiles):
                    for r in sorted(set(re.findall(r'R\d+', smiles)), key=lambda x: int(x[1:])):
                        # Check if there is a format like =Rn, =(Rn), #Rn, or #(Rn)
                        if re.search(rf'(=|#)\({re.escape(r)}\)|{re.escape(r)}', smiles):
                            # If the corresponding format exists, then r should be =Rn, =(Rn), #Rn, or #(Rn)
                            if re.search(rf'=({re.escape(r)})', smiles):
                                r = f'={r}'
                            elif re.search(rf'#({re.escape(r)})', smiles):
                                r = f'#{r}'
                            elif re.search(rf'=\({re.escape(r)}\)', smiles):
                                r = f'={{{r}}}'
                            elif re.search(rf'#\({re.escape(r)}\)', smiles):
                                r = f'#{r}'
                        current_existing_atoms = set(re.findall(r'[FClBrI]', smiles))
                        replaceable_atoms = [atom for atom in possible_atoms if atom not in current_existing_atoms]
                        if replaceable_atoms:
                            chosen_atom = replaceable_atoms[0]
                            smiles = smiles.replace(r, chosen_atom)
                            atom_replacement_map[chosen_atom] = r.replace('=', '').replace('#', '')
                        else:
                            print("No remaining atoms to replace R")
                    return smiles

                new_non_ring_smiles = replace_r_with_atom(non_ring_smiles)

                mol_with_h = Chem.AddHs(Chem.MolFromSmiles(new_non_ring_smiles)) 
                mol_formula = Chem.MolToSmiles(mol_with_h, canonical=True)

                mol_formula = re.sub(r'\[H\]', 'H', mol_formula)
                mol_formula = re.sub(r'\(H\)', 'H', mol_formula)
                mol_formula = re.sub(r'H+', lambda m: f'H{len(m.group(0))}' if len(m.group(0)) > 1 else 'H', mol_formula)
                for old_char, new_char in replace_map.items():
                    mol_formula = mol_formula.replace(old_char, new_char)

                if len(mol_formula) > 1 and mol_formula[1] == '=':
                    mol_formula = mol_formula.replace('*', '')
                else:
                    mol_formula = mol_formula.replace('*', '-')

                for atom, r in atom_replacement_map.items():
                    mol_formula = mol_formula.replace(atom, r)
                mol_formula = mol_formula.replace('#', '≡')
                                   
                output_file_path = os.path.join(folder_path, 'mcs_with_r.txt')
                output_content = f"Nest chain MCS: {mol_formula}\n"
                
                with open(output_file_path, 'a') as f:
                    f.write(output_content)

                # Update mcs_mapping to match the required format
                if mol_formula not in mcs_mapping:
                    mcs_mapping[mol_formula] = {
                        "mcs_structure": mol_formula,
                        "fragments": [],
                        "r_group_mapping": r_group_mapping,
                        'file_name': file_name 
                    }
                mcs_mapping[mol_formula]["fragments"].append({
                    "smiles": original_smiles,
                    "file_name": file_name
                })

        for i, ring_frag in enumerate(ring_fragments):
            original_smiles = Chem.MolToSmiles(ring_frag, canonical=True)           
            cleaned_smiles = clean_smiles_2(original_smiles)
            if isinstance(cleaned_smiles, tuple):
                cleaned_smiles = cleaned_smiles[0] 
            
            drawer = Draw.MolDraw2DCairo(300, 300)
            options = drawer.drawOptions()
            options.useBWAtomPalette()    
            options.colorAtoms = False    
            options.highlightColor = None 
            cleaned_mol = Chem.MolFromSmiles(cleaned_smiles)
            drawer.DrawMolecule(cleaned_mol)
            drawer.FinishDrawing()         
            png_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(png_data))
            img = img.convert("RGB")

            # Save images and SMILES files
            for r in r_labels_set:  
                r_folder_path = os.path.join(folder_path, r)
                os.makedirs(r_folder_path, exist_ok=True)

                img_path = os.path.join(r_folder_path, f'{file_name}_nest_{mcs_counter}.png')
                img.save(img_path)

                smiles_file_path = os.path.join(r_folder_path, f'{file_name}_nest_{mcs_counter}.smiles')
                with open(smiles_file_path, 'w') as smiles_file:
                    smiles_file.write(cleaned_smiles)

        mcs_counter += 1

    global_r_group_counter = max_r_in_fragments + 1
    return handle_fragments, mcs_mapping


def mark_nest_mcs_with_r(mcs_mapping, output_folder):
    global global_r_group_counter
    global global_x_group_counter
    global global_z_group_counter

    initial_r_group_counter = global_r_group_counter
    max_r_in_mcs = initial_r_group_counter
    marked_r_groups = []
    for i, (mcs_smarts, mcs_data) in enumerate(mcs_mapping.items(), start=1):
        global_r_group_counter = initial_r_group_counter

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
                        r_group_mapping[r_atom_idx] = f'R{global_r_group_counter}'
                        marked_r_groups.append(f'R{global_r_group_counter}')
                        global_r_group_counter += 1
                    r_group_atom.SetProp('atomLabel', r_group_mapping[r_atom_idx])
                    for neighbor in r_group_atom.GetNeighbors():
                        atom_indices.add(neighbor.GetIdx())
                else:
                    new_r_group_atom = Chem.Atom(0) 
                    new_r_group_idx = mcs_with_r.AddAtom(new_r_group_atom)
                    mcs_with_r.AddBond(r_atom_idx, new_r_group_idx, Chem.BondType.SINGLE)
                    r_group_mapping[new_r_group_idx] = f'R{global_r_group_counter}'
                    marked_r_groups.append(f'R{global_r_group_counter}')
                    mcs_with_r.GetAtomWithIdx(new_r_group_idx).SetProp('atomLabel', r_group_mapping[new_r_group_idx])
                    atom_indices.add(r_atom_idx)
                    global_r_group_counter += 1
                    max_r_in_mcs = max(max_r_in_mcs, global_r_group_counter)

                    atom_info = {
                        "Index": r_atom_idx,
                        "Symbol": r_group_atom.GetSymbol(),
                        "Degree": r_group_atom.GetDegree(),
                    }

                    if atom_info['Symbol'] in ['N', 'S', 'P', 'O']:
                        print(f"Checking atom: {atom_info}")
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
            print(f"Skipping image generation for MCS with SMARTS containing 'R': {mcs_smarts}")
            continue
        
        # Here is the fix for properly generating and saving images
        asterisk_indices = [atom.GetIdx() for atom in mcs_with_r.GetAtoms() if atom.GetSymbol() == '*']
        r_indices = set(r_group_mapping.keys()) 
        true_asterisk_indices = [idx for idx in asterisk_indices if idx not in r_indices]  

        processed_asterisk_removal = False  

        if len(true_asterisk_indices) >= 2:
            for idx in sorted(true_asterisk_indices, reverse=True):
                bonds = list(mcs_with_r.GetAtomWithIdx(idx).GetBonds())
                for bond in bonds:
                    mcs_with_r.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                mcs_with_r.RemoveAtom(idx)
            processed_asterisk_removal = True  

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

    global_r_group_counter = max_r_in_mcs
    return marked_r_groups
