# main.py
import os
import shutil
import logging
import traceback
from rdkit import Chem
from rdkit.Chem import MolStandardize
import pandas as pd
from src.concrete_content import (
    generate_fragments,
    process_fragments,
    nest_fragments
)
from src.cliams_generate import create_claims
from src.digital import convert_numbers
from src import global_counters

# Configure logging
logging.basicConfig(
    filename='application.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)


def convert_sdf_to_csv(sdf_file, output_folder):
    suppl = Chem.SDMolSupplier(sdf_file)
    data = []

    for mol in suppl:
        if mol is not None:
            try:
                mol = MolStandardize.rdMolStandardize.Cleanup(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
                data.append([smiles])
            except Exception as e:
                logging.warning(f"Failed to convert molecule: {e}")

    if data:
        csv_file = os.path.join(output_folder, os.path.basename(sdf_file).replace(".sdf", ".csv"))
        pd.DataFrame(data, columns=["SMILES"]).to_csv(csv_file, index=False)
        logging.info(f"Converted {sdf_file} to CSV: {csv_file}")
        return csv_file
    else:
        logging.error(f"Conversion failed: No valid molecules in {sdf_file}")
        return None


def remove_empty_folders_upwards(folder_path):
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            remove_empty_folders_upwards(entry_path)
    if not os.listdir(folder_path) or all(f.endswith('.json') for f in os.listdir(folder_path)):
        shutil.rmtree(folder_path)


def run_pipeline(file_path, output_folder):
    global_counters.global_r_group_counter = 1
    global_counters.global_x_group_counter = 1
    global_counters.global_z_group_counter = 1

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    if file_path.endswith('.sdf'):
        csv_file_path = convert_sdf_to_csv(file_path, output_folder)
        if not csv_file_path:
            raise ValueError("Failed to convert SDF to CSV")
        file_path = csv_file_path
    elif not file_path.endswith('.csv'):
        raise ValueError("Only SDF or CSV files are supported")

    total_molecules, r_group_to_atom_info = generate_fragments(file_path, output_folder)
    process_fragments(output_folder)
    nest_fragments(output_folder, r_group_to_atom_info, total_molecules)
    convert_numbers(output_folder, mode="unicode")
    remove_empty_folders_upwards(output_folder)
    create_claims(output_folder)

    return output_folder


if __name__ == "__main__":
    try:
        input_path = input("请输入输入文件路径（SDF 或 CSV）: ").strip()
        output_path = os.path.join(os.path.dirname(input_path), "output")

        result_folder = run_pipeline(input_path, output_path)

    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"An error occurred:\n{error_message}")
        print(f"❌ 错误发生：{e}")



