import os
import re
import shutil


def convert_to_subscript(text, mode="unicode"):
    """
    Convert numbers to subscript format except for certain patterns (like chemical names).

    - The following types of numbers will not be converted:
        - Numbers in chemical compound names (e.g., 1,8-diazabicyclo)
        - Numbers in molecule naming conventions (e.g., 1-pyrrolyl, 4-fluorobenzyl)
    - Other numbers will be converted to subscript.

    Args:
    - text: The input text to be processed.
    - mode: The output mode ("unicode", "latex", or "html").

    Returns:
    - The processed text with appropriate subscript conversions.
    """

    # List of patterns where numbers should NOT be converted to subscript
    exclude_patterns = [
        "1,8-diazabicyclo",
        "1-pyrrolyl",
        "2-methylpyrrolidinyl",
        "3,3-dimethylbutyl",
        "4-dimethylaminopyridine",
        "9-purine-6-amine",
        "4-bromobenzyl",
        "4-chlorobenzyl",
        "4-fluorobenzyl",
        "phenyl substituted by 1-2 of amino",
        "a straight or branched alkylene chain with 1 to 5 carbon atoms",
        "4- to 7-membered ring",
        "1,2-alkylvinylene"
    ]
    
    # Check if the text contains any of the patterns that should not be converted
    for pattern in exclude_patterns:
        if pattern in text:
            return text  # If any of the patterns match, return the text without modification
    
    # If mode is "unicode", use unicode subscript mapping
    if mode == "unicode":
        subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        # Convert numbers to subscript using the map
        return text.translate(subscript_map)
    
    # For Latex or HTML, format differently but same logic
    elif mode == "latex":
        return ''.join([f'$_{{{c}}}$' if c.isdigit() else c for c in text])
    elif mode == "html":
        return ''.join([f'<sub>{c}</sub>' if c.isdigit() else c for c in text])

    return text  # Default return for unrecognized modes


subscript_map = {
    '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9', '₀': '0'
}


def convert_subscript_to_number(part):
    for sub, normal in subscript_map.items():
        part = part.replace(sub, normal)
    return part


def convert_numbers(output_folder, mode="unicode"):
    """
    Convert all numbers after `:` in rule_description.txt and mcs_with_r.txt to subscript format.

    Parameters:
        output_folder (str): The base directory to search for the files.
        mode (str): "unicode" (default), "latex" or "html" to specify the subscript format.
    """
    folder_rename_map = []
    for root, dirs, _ in os.walk(output_folder, topdown=False):
        for dir_name in dirs:
            # Convert the folder name's numbers to subscript
            new_dir_name = convert_to_subscript(dir_name, mode)
            if new_dir_name != dir_name:  # Only rename if the name has changed
                src_path = os.path.join(root, dir_name)
                dst_path = os.path.join(root, new_dir_name)
                
                # Perform the renaming operation
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Rename failed {src_path}: {str(e)}")


    for root, _, files in os.walk(output_folder):
        for filename in ["rule_description.txt", "mcs_with_r.txt"]:
            file_path = os.path.join(root, filename)
            if os.path.exists(file_path):

                with open(file_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()

                modified_lines = []
                for line in lines:
                    if ":" in line:
                        prefix, number_part = line.split(":", 1)
                        number_part = convert_to_subscript(number_part, mode)  
                        modified_line = f"{prefix}:{number_part}"
                    else:
                        modified_line = line 

                    modified_lines.append(modified_line)

                with open(file_path, "w", encoding="utf-8") as file:
                    file.writelines(modified_lines)
