from collections import defaultdict, deque
import io
import os
import json
import shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw, AllChem
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import re
from rdkit import rdBase
from src.mark_r import mark_nest_mcs_with_r, mark_mcs_with_r
from src.fragment import fragment_and_draw
from src.nest import nest_chain, nest_ring
from src.extend import fragment_extension, mol_extension, standardize_smiles, clean_smiles
rdBase.DisableLog('rdApp.*')



def read_data(file_path):
    try:
        df = pd.read_csv(file_path, header=None) 
        return df
    except Exception as e:
        return None


def get_ring_double_bond_signature(mol):
    signature = []
    for bond in mol.GetBonds():
        if bond.IsInRing() and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            a1, a2 = sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            signature.append((a1, a2))
    return tuple(sorted(signature))


def has_three_fused_rings(mol):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    if len(atom_rings) < 3:
        return False

    adjacency = defaultdict(set)
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            if set(atom_rings[i]) & set(atom_rings[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    for start in range(len(atom_rings)):
        if start not in visited:
            queue = deque([start])
            component = []
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    queue.extend(adjacency[node] - visited)
            if len(component) >= 3:
                return True
    return False


def parse_smiles(df, output_folder):
    molecule_list = []
    molecule_index = []
    kekule_groups = defaultdict(list)

    for i, cell in enumerate(df.iloc[:, 0]):
        try:
            mol = Chem.MolFromSmiles(cell)
            if mol:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                signature = get_ring_double_bond_signature(mol)

                molecule_list.append(mol)
                molecule_index.append((i, mol))
                kekule_groups[signature].append(mol)

        except Exception as e:
            print(f"Unable to parse SMILES ({cell}): {e}")

    all_have_three_fused = all(has_three_fused_rings(mol) for mol in molecule_list)

    if all_have_three_fused:
        sorted_groups = sorted(kekule_groups.values(), key=len, reverse=True)
        top_groups = sorted_groups[:2]
        if any(len(group) <= 3 for group in top_groups):
            molecule_mcs = molecule_list
        else:
            molecule_mcs = top_groups[0] if len(top_groups) == 1 else top_groups
    else:
        molecule_mcs = molecule_list

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

    return molecule_list, molecule_index, molecule_mcs


def find_mcs(molecule_list):
    if len(molecule_list) < 2:
        return None
    res = rdFMCS.FindMCS(molecule_list,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                         completeRingsOnly=True) 
    return res.smartsString if res else None


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
                for _ in range(missing_fragments_count):
                    h_atom_img = create_h_atom_image()
                    h_atom_img_path = os.path.join(
                        r_group_path, f"Molecule_{i}_Fragment_H_{current_fragment_count + 1}.png"
                    )
                    h_atom_img.save(h_atom_img_path)

                    # Save SMILES file
                    h_smiles_path = os.path.join(
                        r_group_path, f"Molecule_{i}_Fragment_H_{current_fragment_count + 1}.smiles"
                    )
                    with open(h_smiles_path, "w") as f:
                        f.write("[H]")

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


def check_mcs(mcs_smiles, molecule_mcs):
    mcs_mol = Chem.MolFromSmarts(mcs_smiles)
    if not mcs_mol:
        print("Invalid MCS SMARTS.")
        return mcs_smiles, {}

    heavy_atom_count = sum(1 for atom in mcs_mol.GetAtoms() if atom.GetSymbol() not in ['H', 'D', 'T'])

    if heavy_atom_count <= 3:
        mcs_smiles, differences = find_mcs_2(molecule_mcs)  
        # if mcs_smiles:
        #     mcs_smiles = mcs_smiles.replace("&!@", "").replace("&@", "")
    else:
        differences = {}

    return mcs_smiles, differences


def find_mcs_2(molecule_mcs):
    print (find_mcs_2)
    if len(molecule_mcs) < 2:
        return None, {}

    res = rdFMCS.FindMCS(
        molecule_mcs,
        atomCompare=rdFMCS.AtomCompare.CompareAny,  
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        completeRingsOnly=True
    )

    if res and res.smartsString:
        mcs_smarts = res.smartsString
        differences = record_atom_differences(molecule_mcs, mcs_smarts)
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


def has_fused_ring(mol):
    """Check if the molecule contains fused rings (shared bonds between rings)."""
    if not mol:
        return False
    ri = mol.GetRingInfo()
    
    if not ri or mol.GetNumBonds() == 0:
        return False  
    
    for bond in mol.GetBonds():
        if ri.NumBondRings(bond.GetIdx()) >= 2:
            return True
    return False


def check_mcs_2(mcs_smiles, molecule_mcs):
    """
    Check if all molecules contain fused rings and determine if MCS should be recalculated.
    
    """  
    if not molecule_mcs or any(mol is None for mol in molecule_mcs):
        print("Error: molecule_mcs contains None values.")
        return mcs_smiles, {}

    for mol in molecule_mcs:
        if not isinstance(mol, Chem.Mol):
            return mcs_smiles, {}

    valid_mols = []
    for mol in molecule_mcs:
        if not has_fused_ring(mol):
            return mcs_smiles, {}
        valid_mols.append(mol)

    if len(valid_mols) < 2:
        print("Not enough valid molecules for MCS calculation.")
        return mcs_smiles, {}
    
    mcs_mol = Chem.MolFromSmarts(mcs_smiles)
    if not mcs_mol:
        return None, None, None, None, None
    else:
        Chem.SanitizeMol(mcs_mol)
        if has_fused_ring(mcs_mol):
            return mcs_smiles, {}
    
    try:
        new_mcs, differences = find_mcs_2(valid_mols)  
        if new_mcs:
            # new_mcs = new_mcs.replace("&!@", "").replace("&@", "")
            return new_mcs, differences
    except Exception as e:
        print(f"Error in find_mcs_2: {e}")

    return mcs_smiles, {}


        
def is_valid_mcs(mcs_smiles, molecule_mcs, min_ring_atoms=8):
    """
    Validate if the MCS is structurally trustworthy.
    - If it contains any ring with too many atoms (e.g., >= 8), that ring is considered suspicious.
    - Extract the large ring(s) as individual substructures and check if they appear in ALL molecules.
    """
    try:
        mcs_mol = Chem.MolFromSmarts(mcs_smiles)
        if not mcs_mol:
            print("[MCS Validation] Invalid SMARTS.")
            return False

        Chem.SanitizeMol(mcs_mol)

        ring_info = mcs_mol.GetRingInfo()
        ring_atom_lists = ring_info.AtomRings()
        atom_count = mcs_mol.GetNumAtoms()

        print(f"[MCS Validation] Ring count: {len(ring_atom_lists)}, Atom count: {atom_count}")

        for ring_id, ring_atoms in enumerate(ring_atom_lists):
            if len(ring_atoms) >= min_ring_atoms:
                print(f"[MCS Validation] Detected suspicious large ring (size {len(ring_atoms)}). Verifying its presence in all molecules...")

                # Extract this large ring as a substructure
                large_ring = Chem.PathToSubmol(mcs_mol, ring_atoms)

                # Check if ALL molecules contain this exact ring
                for i, mol in enumerate(molecule_mcs):
                    found = False
                    for mol_ring_atoms in mol.GetRingInfo().AtomRings():
                        subring = Chem.PathToSubmol(mol, mol_ring_atoms)
                        if subring.HasSubstructMatch(large_ring):
                            found = True
                            break
                    if not found:
                        print(f"[MCS Validation] Molecule {i} does NOT contain the large ring.")
                        return False

                print("[MCS Validation] Large ring is confirmed present in all molecules.")
        
        print("[MCS Validation] No suspicious large rings found, or all verified. Accepting MCS.")
        return True

    except Exception as e:
        print(f"[MCS Validation Error] Exception during validation: {e}")
        return False
    
    
def generate_fragments(file_path, output_folder):
    formula_folder = os.path.join(output_folder, 'formula')
    os.makedirs(formula_folder, exist_ok=True)
    
    # Process only a single file
    file_name = os.path.basename(file_path)

    all_fragments = []
    total_molecules = 0
    r_group_to_atom_info = {}

    # Read CSV file
    df = read_data(file_path)
    if df is None:
        print(f"Failed to read data from file: {file_name}")
        return total_molecules, r_group_to_atom_info

    molecule_list, molecule_index, molecule_mcs = parse_smiles(df, output_folder)

    if len(molecule_list) < 2:
        print(f"Not enough molecules for MCS in file: {file_name}")
        return total_molecules, r_group_to_atom_info

    total_molecules = len(molecule_list)

    if isinstance(molecule_mcs[0], list):
        mcs_results = []
        for group in molecule_mcs:
            mcs_smarts = find_mcs(group)
            if mcs_smarts:
                mcs_mol = Chem.MolFromSmarts(mcs_smarts)
                mcs_results.append((mcs_smarts, len(group), mcs_mol.GetNumAtoms()))
            else:
                mcs_results.append((None, len(group), float('inf'))) 

        mcs_results.sort(key=lambda x: (x[2], -x[1])) 

        selected_mcs_smiles, _, _ = mcs_results[0]

        selected_index = mcs_results.index(mcs_results[0])
        molecule_mcs = molecule_mcs[selected_index]

        print(f"[MCS Selection] Selected group {selected_index} with smaller MCS: {selected_mcs_smiles}")
        mcs_smiles = selected_mcs_smiles
    else:
        mcs_smiles = find_mcs(molecule_mcs)
        
    differences = {}
    while True:
        if not mcs_smiles or '?' in mcs_smiles:
            print(f"Could not find a valid MCS: {file_name}")
            return total_molecules, r_group_to_atom_info

        if is_valid_mcs(mcs_smiles, molecule_mcs):
            break

        if len(molecule_mcs) < 2:
            print(f"Insufficient molecules to continue MCS correction: {file_name}")
            return total_molecules, r_group_to_atom_info

        molecule_mcs = molecule_mcs[:len(molecule_mcs)//2]
        mcs_smiles = find_mcs(molecule_mcs)
         
    if not mcs_smiles or '?' in mcs_smiles:
        print(f"Could not find a valid MCS: {file_name}")
        return total_molecules, r_group_to_atom_info
    
    mcs_smiles, differences = check_mcs(mcs_smiles, molecule_mcs)
    mcs_smiles, differences = check_mcs_2(mcs_smiles, molecule_mcs)

    mcs_smiles = mcs_smiles.replace("&!@", "").replace("&@", "")
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
        options.includeAtomNumbers = False
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

        # Generate molecule fragments and images
        r_group_to_atom_info = fragment_and_draw(molecule_list, molecule_index, original_atom_mappings, original_bond_mappings, output_folder)
    else:
        print(f"Failed to convert MCS to molecule object: {mcs_smiles}")

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
    return new_folder_path


def same_fragments(folder_path):
    molecule_fragments = defaultdict(list)

    for file in os.listdir(folder_path):
        if file.endswith('.smiles'):
            try:
                molecule_id = file.split('_')[1]
                with open(os.path.join(folder_path, file), 'r') as f:
                    smiles = f.read().strip() 
                    molecule_fragments[molecule_id].append(smiles)
            except IndexError:
                print(f"Skipping improperly named file: {file}")

    for molecule_id, fragments in molecule_fragments.items():
        # If the fragment contents are not identical, further processing is required
        if len(set(fragments)) > 1:
            return True
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
                print(f"Invalid molecule object: {smiles}")
        except Exception as e:
            print(f"Unable to extract features: {e}")
    return np.array(features)


def read_smiles_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Unable to read file {file_path}: {e}")
        return None


def classify_molecule_1_fragments(folder_path, new_folder_paths):
    """
    Distribute Molecule_1 fragments into different new folders, ensuring each folder contains only one Molecule_1 fragment.
    """

    r_group_folders = new_folder_paths  # Using the new folder paths passed in
    molecule_1_fragments = []

    # Find all Molecule_1 fragments
    for item in os.listdir(folder_path):
        if 'Molecule_1' in item and item.endswith('.smiles'):
            fragment_path = os.path.join(folder_path, item)
            molecule_1_fragments.append(fragment_path)

    if len(molecule_1_fragments) < len(r_group_folders):
        print("Not enough Molecule_1 fragments to distribute into each folder")
        return

    # If there are multiple Molecule_1 fragments, ensure they are in different folders
    for i, fragment_file in enumerate(molecule_1_fragments):
        dest_folder = r_group_folders[i % len(r_group_folders)]  # Ensure distribution into different folders
        dest_smiles_path = os.path.join(dest_folder, os.path.basename(fragment_file))

        # Move the fragment
        shutil.move(fragment_file, dest_smiles_path)

        # Move corresponding PNG file
        img_file = fragment_file.replace('.smiles', '.png')
        if os.path.exists(img_file):
            dest_img_path = os.path.join(dest_folder, os.path.basename(img_file))
            shutil.move(img_file, dest_img_path)


def classify_fragments(folder_path, new_folder_paths):

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
            # print(f"Molecule_{mol_index} fragments are empty, skipping")
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

                img_file = fragment.replace('.smiles', '.png')
                if os.path.exists(img_file):
                    dest_img_path = os.path.join(dest_folder_path, os.path.basename(img_file))
                    shutil.move(img_file, dest_img_path)
            
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
                    continue

            used_folders.add(best_folder)

            # Move fragment and PNG file
            src_path = mol_fragments[mol_idx]
            dest_folder_path = os.path.join(folder_path, best_folder)
            os.makedirs(dest_folder_path, exist_ok=True)

            dest_smiles_path = os.path.join(dest_folder_path, os.path.basename(src_path))
            shutil.move(src_path, dest_smiles_path)

            img_file = src_path.replace('.smiles', '.png')
            if os.path.exists(img_file):
                dest_img_path = os.path.join(dest_folder_path, os.path.basename(img_file))
                shutil.move(img_file, dest_img_path)

            
def verify_r_folders(output_folder, total_molecules, r_groups):
    for r_group in r_groups:
        r_group_path = os.path.join(output_folder, r_group)
        if os.path.isdir(r_group_path):
            fragment_files = [f for f in os.listdir(r_group_path) if f.endswith('.png')]
            if len(fragment_files) != total_molecules:
                print(f"Verification failed for {r_group}. Expected {total_molecules} fragments, found {len(fragment_files)}.")
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


def process_fragments(output_folder):

    # Step 1: Find all folders containing "_"
    folders_to_process = [
        f for f in os.listdir(output_folder)
        if os.path.isdir(os.path.join(output_folder, f)) and "_" in f and len(f.split('_')) > 1
    ]

    # Step 2: Iterate over and process these folders
    for folder in folders_to_process:
        folder_path = os.path.join(output_folder, folder)

        # Check if further processing is needed
        need_processing = same_fragments(folder_path)

        # Step 3: Process based on the same_fragments result
        if not need_processing:
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
        else:
            sub_folders = folder.split('_')  # Split folder name
            new_folder_paths = []
            for sub_folder in sub_folders:
                new_folder_path = new_file(output_folder, sub_folder)  # Create each new folder
                new_folder_paths.append(new_folder_path)

            # Classify into new folders
            classify_molecule_1_fragments(folder_path, new_folder_paths)  # Classify Molecule_1
            classify_fragments(folder_path, new_folder_paths)  # Classify the remaining fragments

    # Step 4: Clean up folders containing "_"
    for folder in folders_to_process:
        folder_path = os.path.join(output_folder, folder)
        # Remove all folders containing "_"
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

    # Step 5: Clean up empty folders
    remove_empty_folders(output_folder)


from src.description import description_data 


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
                                        top_matches = fragment_extension(cleaned_smiles, folder)
                                        for desc in top_matches:  
                                            fragment_identifications.append(desc)
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
                                                    
                                                mol_smiles = Chem.MolToSmiles(mol)
                                                mol_matches = mol_extension(mol_smiles, folder)
                                                for desc in mol_matches:  
                                                    fragment_identifications.append(desc) 
                                                    
                                            else:
                                                fragment_identifications.append(f"{file_name}: No matching description found")
                                        else:
                                            fragment_identifications.append(f"{file_name}: Invalid SMILES format")
                                                                  
                        except Exception as e:
                            error_message = f"Error reading file: {fragment_file_path}, {e}"
                                                    
                if fragment_identifications:
                    processed_folders.append({
                        "folder": folder,
                        "file_names": unmatched_smiles,
                        "identifications": fragment_identifications
                    })
                    
        # Sort processed folders in descending order by the first numeric part in the folder name
        processed_folders.sort(
            key=lambda x: int(subscript_to_int(x["folder"].split('_')[0][1:])), 
            reverse=False
        )

        for entry in processed_folders:
            output_file.write(f"Folder: {entry['folder']}\n")
            for identification in entry["identifications"]:
                output_file.write(f"  {identification}\n")

    return processed_folders


def subscript_to_int(subscript_str):
    """Convert subscript numbers (₀₁₂₃₄₅₆₇₈₉) to normal integers."""
    subscript_digits = "₀₁₂₃₄₅₆₇₈₉"
    normal_digits = "0123456789"
    trans_table = str.maketrans(subscript_digits, normal_digits)
    return subscript_str.translate(trans_table)


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
            continue

        # Extract new valid SMILES
        nest_valid_entries = [
            {'file_name': file_name, 'unmatched_smiles': [s for s in smiles_list if Chem.MolFromSmiles(s)]}
            for file_name, smiles_list in nest_file_names.items()
        ]
        nest_valid_entries = [entry for entry in nest_valid_entries if entry['unmatched_smiles']]

        if not nest_valid_entries:
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
    fragments = []
    mcs_mapping = {}

    for entry in unmatched_smiles_with_filenames:
        file_name = entry['file_name']
        unmatched_smiles = entry['unmatched_smiles'] 

        if not unmatched_smiles:
            continue
        
        for smiles in unmatched_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except Chem.KekulizeException:
                    print(f"Failed to Kekulize molecule: {smiles}")
                    continue
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





