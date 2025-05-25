from collections import Counter, defaultdict
import os
import re
import shutil
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw, AllChem


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

    # Save global fragment information
    identical_fragments_file = os.path.join(output_folder, "identical_fragments.json")
    with open(identical_fragments_file, 'w') as f:
        json.dump(identical_fragments, f, indent=4)

    return r_group_to_atom_info



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
            options.includeAtomNumbers = False

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
            pass
            print(f"Could not split Molecule {molecule_idx+1} into two fragments by breaking bonds connected to Atom {atom_idx}")

    except Exception as e:
        print(f"Failed to fragment molecule {molecule_idx+1} on bond {bond_idx}: {e}")
        pass

    return images, frag_smiles, labels




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


def sanitize_filename(smiles):
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', smiles)
    return sanitized