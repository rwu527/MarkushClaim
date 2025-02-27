import os
import shutil
import logging
import tkinter as tk
from tkinter import simpledialog, messagebox
from rdkit import Chem
import pandas as pd
from concrete_content import (
    generate_fragments,
    process_fragments,
    nest_fragments
)
from cliams_generate import create_pdf_in_current_directory
logging.basicConfig(filename="app.log", level=logging.DEBUG)
logging.debug("Debugging message")


def convert_sdf_to_csv(sdf_file, output_folder):
    """Convert an SDF file to CSV format."""
    suppl = Chem.SDMolSupplier(sdf_file)
    data = []

    for mol in suppl:
        if mol is not None:
            try:
                Chem.Kekulize(mol)
                
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                data.append([smiles])
            except Exception as e:
                logging.warning(f"Failed to convert molecule to SMILES: {e}")

    if data:
        csv_file = os.path.join(output_folder, os.path.basename(sdf_file).replace(".sdf", ".csv"))
        df = pd.DataFrame(data, columns=["SMILES"])
        df.to_csv(csv_file, index=False)
        logging.info(f"Converted {sdf_file} to CSV: {csv_file}")
        return csv_file
    else:
        logging.error(f"Conversion failed for {sdf_file}: No valid molecules found.")
        return None


def remove_empty_folders_upwards(folder_path):
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        if os.path.isdir(entry_path):
            remove_empty_folders_upwards(entry_path)

    entries = os.listdir(folder_path)
    if not entries or (len(entries) == 1 and entries[0].endswith('.json')):
        shutil.rmtree(folder_path)


def main():
    root = tk.Tk()
    root.withdraw() 

    file_path = simpledialog.askstring("Input File Path (Chinese paths are not supported)", "Please enter the full path of the file (e.g., C:/path/to/your/file.sdf or file.csv):")

    if not file_path:
        messagebox.showerror("Error", "No file path entered, program will exit.")
        return

    if not os.path.isfile(file_path):
        messagebox.showerror("Error", f"Error: '{file_path}' is not a valid file path. Please ensure the path is correct.")
        return

    folder_path = os.path.dirname(file_path)

    output_folder = os.path.join(folder_path, 'output')

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    try:
        if file_path.endswith('.sdf'):
            csv_file_path = convert_sdf_to_csv(file_path, output_folder)
            if not csv_file_path:
                messagebox.showerror("Error", f"Failed to convert SDF file to CSV: {file_path}")
                return
            file_path = csv_file_path 
        elif not file_path.endswith('.csv'):
            messagebox.showerror("Error", "Only SDF or CSV files are supported.")
            return

        total_molecules, r_group_to_atom_info = generate_fragments(file_path, output_folder)

        process_fragments(output_folder)
        nest_fragments(output_folder, r_group_to_atom_info, total_molecules)
        remove_empty_folders_upwards(output_folder)

        create_pdf_in_current_directory(output_folder)

        messagebox.showinfo("Completion", f"Processing completed, results have been saved to {output_folder}")
    finally:
        root.after(0, lambda: root.quit()) 

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        messagebox.showerror("Error", "Input file is invalid.")

     
