import io
import itertools
import os
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw
from rdkit.Chem import rdmolops
from src.fragment import sanitize_filename
from PIL import Image
import re
from src import global_counters


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
    '(D)(D)(D)':'D3',
    '(D)(D)':'D2',
    '3H': 'T',
    'COH': 'CHO',
    '(CH3)(CH3)CH3': '(CH3)3',
    '(CD3)(CD3)CD3': '(CD3)3',
    '(CH3)CH3': '(CH3)2',
    '(CD3)CD3': '(CD3)2',
    'CH2CH2CH2CH2CH2CH2CH2CH2CH2CH2CH2CH2': '(CH2)12',
    'CH2CH2CH2CH2CH2CH2CH2CH2CH2CH2CH2': '(CH2)11',
    'CH2CH2CH2CH2CH2CH2CH2CH2CH2CH2': '(CH2)10',
    'CH2CH2CH2CH2CH2CH2CH2CH2CH2': '(CH2)9',
    'CH2CH2CH2CH2CH2CH2CH2CH2': '(CH2)8',
    'CH2CH2CH2CH2CH2CH2CH2': '(CH2)7',
    'CH2CH2CH2CH2CH2CH2': '(CH2)6',
    'CH2CH2CH2CH2CH2': '(CH2)5',
    'CH2CH2CH2CH2': '(CH2)4',
    'CH2CH2CH2': '(CH2)3',
    'CH2CH2': '(CH2)2',
    'C≡N': 'CN', 
    'CH=O': 'CHO',
    "N=N=N":'N3'
}



def is_simple_mcs(mcs_mol):
    atom_count = mcs_mol.GetNumAtoms()
    heavy_atom_count = sum(1 for atom in mcs_mol.GetAtoms() if atom.GetAtomicNum() != 1)  
    if heavy_atom_count <= 2:
        return True
    return False


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


import random


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
                smiles = Chem.MolToSmiles(mol)
                if 'oc(SC)' in smiles:
                    continue
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

        # Step 5: Try different pairs of fragments and compute MCS
        success = False
        all_combinations = list(itertools.combinations(kekulized_mols, 2))  # Generate all pairs

        # Keep track of the attempt count to avoid infinite loops
        attempt_count = 0
        max_attempts = len(all_combinations)

        while not success and attempt_count < max_attempts:
            random.shuffle(all_combinations)  # Shuffle combinations to randomize pair selection
            for mol1, mol2 in all_combinations:
                try:
                    mcs_result = rdFMCS.FindMCS([mol1, mol2], parameters=mcs_params)
                except Exception:
                    continue  # If computation fails, continue to the next pair

                if mcs_result.canceled:
                    continue  # Skip canceled results

                # Check if the MCS contains '[#0]' (wildcard) and is valid
                mcs_smarts = mcs_result.smartsString
                if '[#0]' in mcs_smarts:  # Valid MCS found
                    mcs_mol = Chem.MolFromSmarts(mcs_smarts)
                    Chem.Kekulize(mcs_mol, clearAromaticFlags=True)
                    mcs_smiles = Chem.MolToSmiles(mcs_mol)  

                    if mcs_mol:
                        mcs_smiles = extract_core_ring(mcs_mol)
                        
                    # Record the successful MCS in mcs_mapping
                    mcs_mapping[mcs_smiles] = {
                        'mcs_structure': mcs_smiles,
                        'fragments': []
                    }

                    # Collect SMILES and file names for all fragments in the group
                    for frag in group_fragments:
                        frag_smiles = frag.get('smiles', 'Unknown_SMILES')
                        frag_file_name = frag.get('file_name', 'Unknown_File')

                        mcs_mapping[mcs_smiles]["fragments"].append({
                            "smiles": frag_smiles,
                            "file_name": frag_file_name
                        })

                    success = True
                    break  # Exit the loop if a successful MCS is found

            attempt_count += 1

        # Step 6: If no valid MCS was found, mark all fragments as failed
        if not success:
            for frag in group_fragments:
                failed_fragments.append(frag)

     
    # # Step 7: Check and update MCS SMILES for missing '*' atoms
    # for mcs_smiles, group_info in mcs_mapping.items():
    #     if "*" not in mcs_smiles:
    #         mcs_mapping = generate_mcs_from_ring(group_info['fragments'], mcs_mapping)        
    generate_nest_image(mcs_mapping, folder_path, prefix="mcs_ring")
                
    if failed_fragments:
        for frag in failed_fragments:
            smiles = frag.get('smiles', 'Unknown_SMILES')
            cleaned_smiles = clean_smiles_2(smiles)[0]
            sanitized_smiles = sanitize_filename(cleaned_smiles)
            mol = Chem.MolFromSmiles(cleaned_smiles)
            Chem.Kekulize(mol, clearAromaticFlags=True)

            if mol:
                # ===== Use enhanced ring extractor instead of manual filtering =====
                ring_only_smiles = extract_core_ring(mol)
                ring_only_mol = Chem.MolFromSmiles(ring_only_smiles)

                # ===== Generate and save image =====
                drawer = Draw.MolDraw2DCairo(300, 300)
                options = drawer.drawOptions()
                options.useBWAtomPalette()
                options.colorAtoms = False
                options.highlightColor = None
                drawer.DrawMolecule(ring_only_mol)
                drawer.FinishDrawing()

                png_data = drawer.GetDrawingText()
                img = Image.open(io.BytesIO(png_data))
                img = img.convert("RGB")

                img_path = os.path.join(folder_path, f'failed_{sanitized_smiles}.png')
                img.save(img_path)

                # ===== Add to mcs_mapping =====
                mcs_mapping[ring_only_smiles] = {
                    'mcs_structure': ring_only_smiles,
                    'fragments': [{
                        "smiles": frag['smiles'],
                        "file_name": frag['file_name']
                    }]
                }
    mcs_mapping = sort_mcs_mapping(mcs_mapping)                
    print(mcs_mapping)  
      
    return fragments, mcs_mapping


def sort_mcs_mapping(mcs_mapping):
    """
    Sorts the mcs_mapping dictionary by the core structure of each mcs_smiles,
    where the core is defined as the mcs_smiles with '*' characters removed.
    
    This reorders entries so that structurally equivalent cores (ignoring attachment points)
    are grouped together in the output. The structure and values remain unchanged.
    """

    grouped = defaultdict(list)

    # Group by core (mcs_smiles without asterisks)
    for mcs_smiles, data in mcs_mapping.items():
        core_smiles = mcs_smiles.replace('*', '')
        grouped[core_smiles].append((mcs_smiles, data))

    # Sort by core SMILES string and reconstruct the mapping in that order
    sorted_mapping = {}
    for core in sorted(grouped.keys()):
        for mcs_smiles, data in grouped[core]:
            sorted_mapping[mcs_smiles] = data  # Preserve original keys and values

    return sorted_mapping



def extract_core_ring(mcs_mol):
    """ Optimized core ring extraction function """
    try:
        # Save the original SMILES for exception handling
        original_smiles = Chem.MolToSmiles(mcs_mol)
        
        # ===== Pre-check for atom deletion necessity =====
        # Check if there are non-ring atoms that need to be removed (excluding * atoms)
        has_sidechain = any(
            (not atom.IsInRing() and atom.GetSymbol() != '*')
            for atom in mcs_mol.GetAtoms()
        )
        if not has_sidechain:
            return original_smiles  # Return directly if there is no sidechain

        # ===== Core processing workflow =====
        ed_mol = Chem.EditableMol(mcs_mol)
        # Atom retention flag (keep ring atoms and * atoms)
        keep_flags = [atom.IsInRing() or atom.GetSymbol() == '*' 
                      for atom in mcs_mol.GetAtoms()]
        
        # ===== Step 1: Identify the atoms in the rings =====
        # Get ring information from the molecule
        ring_info = mcs_mol.GetRingInfo().AtomRings()
        
        # If no rings, return original structure
        if not ring_info:
            return original_smiles
        
        # ===== Step 2: Keep all atoms in the rings =====
        # Mark atoms involved in the rings
        for ring in ring_info:
            for atom_idx in ring:
                atom = mcs_mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    nbr_idx = neighbor.GetIdx()
                    if keep_flags[nbr_idx]:
                        continue
                    bond = mcs_mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
                    if bond and bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]:
                        keep_flags[nbr_idx] = True

        # ===== Step 3: Expand retention to connected atoms (including * atoms) =====
        visited = set()
        queue = [i for i, flag in enumerate(keep_flags) if flag]
        while queue:
            idx = queue.pop(0)
            if idx in visited:
                continue
            visited.add(idx)
            
            atom = mcs_mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == '*':
                for nbr in atom.GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    if not keep_flags[nbr_idx]:
                        keep_flags[nbr_idx] = True
                        queue.append(nbr_idx)

        # ===== Step 4: Remove non-retained atoms in reverse order =====
        remove_atoms = [i for i, flag in enumerate(keep_flags) if not flag]
        if not remove_atoms:  # Second check for deletion necessity
            return original_smiles
        
        for idx in sorted(remove_atoms, reverse=True):
            ed_mol.RemoveAtom(idx)
        
        # ===== Post-processing =====
        core_mol = ed_mol.GetMol()
        # Validity check
        if core_mol.GetNumAtoms() == 0:
            return original_smiles
        
        # Get the largest connected fragment (if multiple fragments exist)
        frags = rdmolops.GetMolFrags(core_mol, asMols=True)
        largest_frag = max(frags, key=lambda x: x.GetNumAtoms(), default=None)
        
        return Chem.MolToSmiles(largest_frag) if largest_frag else original_smiles

    except Exception as e:  # Catch all exceptions
        print(f"Error in extract_core_ring: {str(e)}")
        return original_smiles


def generate_mcs_from_ring(group_fragments, mcs_mapping):
    for mcs_smiles, group_info in mcs_mapping.items():

        # Store the extraction results for each fragment
        extracted_smiles_list = []

        # Get the SMILES for each molecular fragment
        for frag in group_info['fragments']:
            smiles = frag.get('smiles', '')
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            try:
                # Find all atoms marked as '*'
                star_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
                
                if not star_atoms:
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
                continue

        # If the extracted results list is empty, skip this group
        if not extracted_smiles_list:
            continue

        # Count the extracted results and select the most frequent SMILES
        smiles_counter = Counter(extracted_smiles_list)
        most_common_smiles = smiles_counter.most_common(1)[0][0]
        most_common_smiles = re.sub(r'\[\d+\*]', '*', most_common_smiles)

        mcs_mapping[mcs_smiles]["mcs_structure"] = most_common_smiles

    return mcs_mapping


def nest_chain(fragments, folder_path):

    mcs_counter = 1
    mcs_mapping = {}
    handle_fragments = {}
    r_labels_set = set()  # Store unique R labels

    max_r_in_fragments = global_counters.global_r_group_counter  

    for fragment in fragments:
        mol = fragment['mol']
        original_smiles = fragment['smiles']
        file_name = fragment.get('file_name', 'Unknown')  

        cleaned_smiles, bond_marker_num = clean_smiles_1(original_smiles)
        mol = Chem.MolFromSmiles(cleaned_smiles)

        if not mol:
            print(f"Error: Invalid cleaned SMILES: {cleaned_smiles}")
            continue

        bond_marker_idx = None

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == '*':
                bond_marker_idx = atom.GetIdx()
                break

        if bond_marker_idx is None:
            continue

        # print(f"Processing fragment: {original_smiles}")
        # print(f"Cleaned SMILES: {cleaned_smiles}")
        # print(f"Bond Marker Index: {bond_marker_idx}")

        visited_atoms = set()
        bonds_to_break = set()
        r_counter = global_counters.global_r_group_counter

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

        if not bonds_to_break:
            # print(f"Error: No bonds found for fragmentation in {cleaned_smiles}, skipping...")
            continue

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
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
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

    global_counters.global_r_group_counter = max_r_in_fragments + 1
    return handle_fragments, mcs_mapping