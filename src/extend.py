from src.extend_frag import fragment_data  
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from src.description import description_data 


def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
        return standardized_smiles
    return None


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


def get_sparse_fingerprint(smiles, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    fp = AllChem.GetMorganFingerprint(mol, radius)
    fp_dict = fp.GetNonzeroElements()
    fp_str = ",".join(f"{bit}:{count}" for bit, count in fp_dict.items())
    return fp_str


def parse_fingerprint(fp_str):
    fp_dict = {}
    if not fp_str:
        return fp_dict
    for pair in fp_str.split(','):
        if not pair:
            continue
        bit, count = pair.split(':')
        fp_dict[int(bit)] = int(count)
    return fp_dict


def calculate_tanimoto(target_fp, query_fp):
    common_bits = set(target_fp.keys()) & set(query_fp.keys())
    intersection = sum(min(target_fp[bit], query_fp[bit]) for bit in common_bits)
    sum_target = sum(target_fp.values())
    sum_query = sum(query_fp.values())
    denominator = sum_target + sum_query - intersection
    
    if denominator == 0:
        return 0.0
    return intersection / denominator


def fragment_extension(cleaned_smiles, folder):
    if cleaned_smiles == "[H]":
        return []

    try:
        target_fp_str = get_sparse_fingerprint(cleaned_smiles)
        target_fp = parse_fingerprint(target_fp_str)
    except ValueError as e:
        print(f"Error processing SMILES: {e}")
        return []

    similarities = []
    for frag_smiles, frag_fp_str in fragment_data.items():
        frag_fp = parse_fingerprint(frag_fp_str)
        similarity = calculate_tanimoto(target_fp, frag_fp)
        similarities.append((similarity, frag_smiles))

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_results = similarities[:20]

    results = []
    matched_fragments = set(smiles for _, smiles in top_results)

    def find_description(smiles):
        for entry in description_data:
            smiles_list = []
            for sm in entry.get("SMILE", "").split(";"):
                cleaned_smile = standardize_smiles(clean_smiles(sm))
                if cleaned_smile is not None:
                    smiles_list.append(cleaned_smile)
                else:
                    smiles_list.append(sm)

            if smiles in smiles_list:
                return entry.get("Description")
        return None

    for _, smiles in top_results:
        description = find_description(smiles)
        if description:
            results.append(f"Molecule_0_extend.smiles: {description}")

    special_cases = {"*C", "*CC", "*C(C)C", "*CCC"}
    if cleaned_smiles in special_cases and "*C1CC1" not in matched_fragments:
        description = find_description("*C1CC1")
        if description:
            results.append(f"Molecule_0_extend.smiles: {description}")

    return results


def mol_extension(cleaned_smiles, folder):
    if cleaned_smiles == "[H]":
        return []

    extension_map = {
        "*C#N": "*C(=O)OC(C)C",
        "*C(=O)OC": "*C#N",
        "*N#C": "*OC(=O)C",
    }

    smiles_to_process = [(cleaned_smiles, "original")]
    if cleaned_smiles in extension_map:
        smiles_to_process.append((extension_map[cleaned_smiles], "extended"))

    results = []
    seen_formulas = set()  

    for query_smiles, tag in smiles_to_process:
        try:
            target_fp_str = get_sparse_fingerprint(query_smiles)
            target_fp = parse_fingerprint(target_fp_str)
        except ValueError as e:
            print(f"Error processing SMILES {query_smiles}: {e}")
            continue

        similarities = []
        for frag_smiles, frag_fp_str in fragment_data.items():
            frag_fp = parse_fingerprint(frag_fp_str)
            similarity = calculate_tanimoto(target_fp, frag_fp)
            similarities.append((similarity, frag_smiles))

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_results = similarities[:20]

        for _, smiles in top_results:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                ring_info = mol.GetRingInfo()
                rings = ring_info.AtomRings()

                if not rings:
                    mol_with_h = Chem.AddHs(mol)
                    mol_formula = Chem.MolToSmiles(mol_with_h, canonical=True)

                    mol_formula = re.sub(r'\[H\]', 'H', mol_formula)
                    mol_formula = re.sub(r'\(H\)', 'H', mol_formula)
                    mol_formula = re.sub(r'H+', lambda m: f'H{len(m.group(0))}' if len(m.group(0)) > 1 else 'H', mol_formula)

                    for old_char, new_char in replace_map.items():
                        mol_formula = mol_formula.replace(old_char, new_char)

                    if mol_formula.startswith('='):
                        continue
                    else:
                        mol_formula = f"-{mol_formula}"

                    if mol_formula not in seen_formulas:
                        seen_formulas.add(mol_formula)
                        results.append(f"Molecule_0_extend.smiles: {mol_formula} ")

    return results


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