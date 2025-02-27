
import logging
import os
import random
import re
import time
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageEnhance
logging.basicConfig(filename='mcs_finder.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

PAGE_WIDTH, PAGE_HEIGHT = letter
LINE_HEIGHT = 12 * 1.5
MARGIN_LEFT = 72
MARGIN_RIGHT = 72
MARGIN_TOP = 750
PAGE_BOTTOM = 50

def create_empty_pdf(pdf_path, output_folder):
    print(f"Creating PDF at: {pdf_path}")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    y_position = MARGIN_TOP
    c.setFont("Times-Bold", 12)
    c.drawString(MARGIN_LEFT, y_position, "Claims")
    y_position -= LINE_HEIGHT * 1.5
    c.setFont("Times-Bold", 12)
    c.drawString(MARGIN_LEFT, y_position, "1.")
    c.setFont("Times-Roman", 12)
    c.drawString(MARGIN_LEFT + 18, y_position, "A compound of formula (I)")

    # Set the formula folder path to "output/formula" in the output directory
    formula_folder = os.path.join(output_folder, 'formula')
    # Ensure the output formula folder exists
    if not os.path.exists(formula_folder):
        os.makedirs(formula_folder)
    image_files = [f for f in os.listdir(formula_folder) if f.endswith('.png')]
    if image_files:
        image_path = os.path.join(formula_folder, image_files[0])

        img = Image.open(image_path)
        
        # Check aspect ratio to decide scaling
        img_width, img_height = img.size
        if img_width >= img_height:
            scaled_width = 150
            scaled_height = img_height * (scaled_width / img_width)
        else:
            scaled_height = 150
            scaled_width = img_width * (scaled_height / img_height)
        
        # Save the original image
        img.save(os.path.join(formula_folder, "temp_image.png"))
        
        # Set image position
        img_path = ImageReader(os.path.join(formula_folder, "temp_image.png"))
        x_position = (PAGE_WIDTH - scaled_width) / 2
        y_position -= LINE_HEIGHT + scaled_height + 10  # Extra space below the image
        c.drawImage(img_path, x_position, y_position, width=scaled_width, height=scaled_height)

        # Insert (I) below the image
        y_position -= 15
        c.drawString(x_position + scaled_width / 2 - 5, y_position, "(I)")
        os.remove(os.path.join(formula_folder, "temp_image.png"))
        y_position -= LINE_HEIGHT * 2  # Extra line spacing
    
    # Left-aligned "wherein"
    c.setFont("Times-Roman", 12)
    c.drawString(MARGIN_LEFT, y_position, "wherein:")
    y_position -= LINE_HEIGHT

    # Get all subfolder names
    subfolders = [
        f for f in os.listdir(output_folder)
        if os.path.isdir(os.path.join(output_folder, f)) and f != 'formula'
    ]

    # To record processed R groups and avoid duplication
    processed_r_groups = set()
    all_valid_parts = []

    # Step 1: Extract all valid parts
    for folder_name in subfolders:
        # If the folder name contains "_"
        if "_" in folder_name:
            parts = folder_name.split("_")  # Split by "_"
            logging.debug(f"Splitting folder '{folder_name}' into parts: {parts}")
        else:
            parts = [folder_name] 
        for part in parts:
            if part.startswith("R") and part not in processed_r_groups:
                all_valid_parts.append(part)
                processed_r_groups.add(part)
            elif not part.startswith("R"):
                all_valid_parts.append(part)

    logging.debug(f"Before sorting, all valid parts: {all_valid_parts}")
    sorted_parts = sorted(
        all_valid_parts,
        key=lambda x: int(re.search(r"\d+", x).group()) if x.startswith("R") and re.search(r"\d+", x) else float('inf')
    )
    logging.debug(f"After sorting, sorted parts: {sorted_parts}")

    for i, part in enumerate(sorted_parts):
        y_position = add_folder_description(
            c,
            part,
            os.path.join(output_folder, part),
            y_position,
            is_last_folder=(i == len(sorted_parts) - 1)
        )
        if y_position < PAGE_BOTTOM:
            c.showPage()
            y_position = MARGIN_TOP
            c.setFont("Times-Roman", 12)

    c.save()
    remove_blank_pages(pdf_path)


def remove_blank_pages(pdf_path):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    for page in reader.pages:
        text = page.extract_text()
        if text.strip(): 
            writer.add_page(page)

    with open(pdf_path, 'wb') as output_pdf:
        writer.write(output_pdf)

    print(f'Created PDF without blank pages: {pdf_path}')
    

def process_nest_folder(c, part, folder_path, y_position):
    R_nest_description = ""

    # Get all subfolders in the given folder path
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    nest_folder_path = None
    # Search for the _nest subfolder corresponding to the part
    for subfolder in subfolders:
        if part in subfolder and "_nest" in subfolder:
            nest_folder_path = os.path.join(folder_path, subfolder)
            break                    

    if nest_folder_path and os.path.isdir(nest_folder_path):
        # Get all mcs_with_r_ images in the nest folder
        mcs_images = [f for f in os.listdir(nest_folder_path) if f.startswith("mcs_with_r_") and f.lower().endswith(".png")]

        all_images = mcs_images       
        image_paths = []
        if all_images:
            for index, image_file in enumerate(all_images):
                image_path = os.path.join(nest_folder_path, image_file)
                image_paths.append(image_path)

            print(f"[Debug] Returning images: {image_paths}") 

        # Also look for 0_mcs_with_r_ images in the nest folder
        form_images = [f for f in os.listdir(nest_folder_path) if f.startswith("0_mcs_with_r_") and f.lower().endswith(".png")]

        form_paths = []
        if form_images:
            for index, image_file in enumerate(form_images):
                form_path = os.path.join(nest_folder_path, image_file)
                form_paths.append(form_path)

            print(f"[Debug] Returning form images: {form_paths}")

        # Check if the rule_description.txt exists and process it
        txt_file_path = os.path.join(nest_folder_path, "mcs_with_r.txt")
        if os.path.exists(txt_file_path):
            print(f"[Debug] Found txt file: {txt_file_path}")  
                
            with open(txt_file_path, "r") as file:
                lines = file.readlines()
                nest_chain_mcs = []
                    
                for line in lines:
                    if line.startswith("Nest chain MCS:"):
                        content = line.split("Nest chain MCS:")[1].strip()
                        nest_chain_mcs.extend(content.split(", "))

                unique_mcs = list(set(nest_chain_mcs))
                unique_mcs.sort()  
                unique_mcs = ["-" + mcs for mcs in unique_mcs]
                expanded_mcs = []
                for mcs in unique_mcs:
                    mcs_expanded = mcs 

                    possible_expansions = []
                    
                    if 'CH2' in mcs:
                        possible_expansions.append("remove_CH2")
                        possible_expansions.append("expand_CH2")                    
                    if not mcs.startswith('CH2'):
                        possible_expansions.append("add_CH2_at_start")
                    if 'NH' in mcs or 'SO2' in mcs:
                        possible_expansions.append("replace_S_N")
                    expansion_type = random.choice(possible_expansions)

                    if expansion_type == "remove_CH2":
                        mcs_expanded = mcs_expanded.replace('CH2', '', 1)  
                    elif expansion_type == "expand_CH2":
                        mcs_expanded = mcs_expanded.replace('CH2', '(CH2)2', 1)  
                    elif expansion_type == "add_CH2_at_start":
                        dash_index = mcs_expanded.find('-')
                        if dash_index != -1:
                            mcs_expanded = mcs_expanded[:dash_index + 1] + 'CH2' + mcs_expanded[dash_index + 1:]
                        else:
                            mcs_expanded = 'CH2' + mcs_expanded
                    elif expansion_type == "replace_S_N":
                        if 'N' in mcs_expanded:
                            mcs_expanded = mcs_expanded.replace('N', 'S', 1)
                        elif 'S' in mcs_expanded:
                            mcs_expanded = mcs_expanded.replace('S', 'N', 1)
                    mcs_expanded = mcs_expanded.replace('-=', '=')
                    mcs_expanded = mcs_expanded.replace('-≡', '≡')
                    mcs_expanded = mcs_expanded.replace('CH2CH2', '(CH2)2')                             
                    expanded_mcs.append(mcs)  
                    expanded_mcs.append(mcs_expanded) 

                R_nest_description = ", ".join(expanded_mcs[:-1]) + " or " + expanded_mcs[-1] + ";"
            print(f"[Debug] Returning description: {R_nest_description}") 
        return y_position, nest_folder_path, True, R_nest_description, image_paths, form_paths

    return y_position, nest_folder_path, False, R_nest_description, [], []


def generate_images(c, image_paths, y_position):
    """
    Function to generate images later, using the returned image paths.
    This function will handle the scaling, positioning, and drawing of images
    on the PDF after the process_nest_folder function has been called.
    """
    for index, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert("L")  
        img = ImageEnhance.Contrast(img).enhance(2.0)  
        img = img.crop(img.getbbox())

        img_width, img_height = img.size
        if img_width >= img_height:
            scaled_width = 100  
            scaled_height = img_height * (scaled_width / img_width)
        else:
            scaled_height = 100  
            scaled_width = img_width * (scaled_height / img_height)
        bw_img = ImageReader(img)
        if y_position - (LINE_HEIGHT + scaled_height + 5) < PAGE_BOTTOM:
            c.showPage()  
            y_position = MARGIN_TOP  
            c.setFont("Times-Roman", 12) 

        if index % 3 == 0:  # Left image
            x_position = MARGIN_LEFT + (PAGE_WIDTH / 3 - MARGIN_LEFT) / 2 - (scaled_width / 2) + 20
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)
        elif index % 3 == 1:  # Center image
            x_position = PAGE_WIDTH / 2 - scaled_width / 2
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)
        else:  # Right image
            x_position = PAGE_WIDTH - MARGIN_LEFT - scaled_width - 30
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)

        text_x_position = x_position + scaled_width  
        text_y_position = y_position + 30 - scaled_height  
        if index == len(image_paths) - 2: 
            c.drawString(text_x_position, text_y_position, "  or")
        elif index == len(image_paths) - 1:
            c.drawString(text_x_position, text_y_position, ";")
        else:  
            c.drawString(text_x_position, text_y_position, ",")
        if index % 3 == 2 or index == len(image_paths) - 1:
            y_position -= (scaled_height + 5)                      

    return y_position

def generate_images_2(c, image_paths, y_position):
    """
    Function to generate images later, using the returned image paths.
    This function will handle the scaling, positioning, and drawing of images
    on the PDF after the process_nest_folder function has been called.
    """
    for index, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert("L")  
        img = ImageEnhance.Contrast(img).enhance(2.0)  
        img = img.crop(img.getbbox())

        img_width, img_height = img.size
        if img_width >= img_height:
            scaled_width = 100  
            scaled_height = img_height * (scaled_width / img_width)
        else:
            scaled_height = 100  
            scaled_width = img_width * (scaled_height / img_height)
        bw_img = ImageReader(img)
        if y_position - (LINE_HEIGHT + scaled_height + 5) < PAGE_BOTTOM:
            c.showPage()  
            y_position = MARGIN_TOP  
            c.setFont("Times-Roman", 12) 

        if index % 3 == 0:  # Left image
            x_position = MARGIN_LEFT + (PAGE_WIDTH / 3 - MARGIN_LEFT) / 2 - (scaled_width / 2) + 20
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)
        elif index % 3 == 1:  # Center image
            x_position = PAGE_WIDTH / 2 - scaled_width / 2
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)
        else:  # Right image
            x_position = PAGE_WIDTH - MARGIN_LEFT - scaled_width - 30
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)

        text_x_position = x_position + scaled_width  
        text_y_position = y_position + 30 - scaled_height  

        c.drawString(text_x_position, text_y_position, ",")
        if index % 3 == 2 or index == len(image_paths) - 1:
            y_position -= (scaled_height + 5)                      

    return y_position


def generate_form_images(c, image_paths, y_position):
    """
    Function to generate images later, using the returned image paths.
    This function will handle the scaling, positioning, and drawing of images
    on the PDF after the process_nest_folder function has been called.
    """
    for index, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert("L")  
        img = ImageEnhance.Contrast(img).enhance(2.0)  
        img = img.crop(img.getbbox())

        img_width, img_height = img.size
        if img_width >= img_height:
            scaled_width = 100  
            scaled_height = img_height * (scaled_width / img_width)
        else:
            scaled_height = 100  
            scaled_width = img_width * (scaled_height / img_height)
        bw_img = ImageReader(img)
        if y_position - (LINE_HEIGHT + scaled_height + 5) < PAGE_BOTTOM:
            c.showPage()  
            y_position = MARGIN_TOP  
            c.setFont("Times-Roman", 12) 

        if index % 3 == 0:  # Left image
            x_position = MARGIN_LEFT + (PAGE_WIDTH / 3 - MARGIN_LEFT) / 2 - (scaled_width / 2) + 20
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)
        elif index % 3 == 1:  # Center image
            x_position = PAGE_WIDTH / 2 - scaled_width / 2
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)
        else:  # Right image
            x_position = PAGE_WIDTH - MARGIN_LEFT - scaled_width - 30
            c.drawImage(bw_img, x_position, y_position-scaled_height+10, width=scaled_width, height=scaled_height)

        text_x_position = x_position + scaled_width  
        text_y_position = y_position + 30 - scaled_height  
        c.drawString(text_x_position, text_y_position, ",")
        if index % 3 == 2 or index == len(image_paths) - 1:
            y_position -= (scaled_height + 5)                      

    return y_position

seed_initialized = False  
random_choice = None  

def process_rule_description(part, folder_path):
    """Process the rule description file and return a combined description."""
    global seed_initialized, random_choice
    rule_description_path = os.path.join(os.path.dirname(folder_path), "rule_description.txt")
    combined_description = "" 
    forms_description = ""

    # List of fallback items to choose from when items_sorted_all is empty
    fallback_items = [
        "an unsubstituted C6-C8 heterocyclic ring containing two oxygen atoms and one sulfur atom",
        "a substituted C5-C6 cycloalkyl ring optionally containing one nitrogen atom",
        "a substituted C5-C6 cycloalkyl ring optionally containing one oxygen atom",
        "a substituted C5-C7 cycloalkyl ring",
        "a substituted C4-C5 heteroaryl ring optionally containing 1-2 heteroatoms independently selected from N, O, or S",
        "a substituted C5-C7 cycloalkyl ring substituted with 1-2 fluoro groups",
        "an unsubstituted C6 heteroaryl ring substituted with one methoxy group",
        "a substituted C5-C6 cycloalkyl ring substituted with a combination of methyl, chloro, and nitro groups",
        "a substituted fused bicyclic aryl system comprising a benzene ring and a pyridine ring"
    ]

    # If the seed has not been initialized, initialize it once
    if not seed_initialized:
        random.seed(time.time())  # Use the current timestamp as the seed to ensure a different seed each time the program runs
        random_choice = random.choice(fallback_items)  # Choose a random item and save it
        seed_initialized = True  # Set to initialized

    if os.path.exists(rule_description_path):
        print(f"Found rule description file: {rule_description_path}")

        with open(rule_description_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            descriptions = set()
            ring_descriptions = set()
            in_folder_section = False
            other_rs = set()

            for line in lines:
                # Check if the content after "Folder:" contains "_", if it does, skip it
                if line.startswith("Folder:") and part in line:
                    folder_info = line.strip().split(":")[1].strip() if ":" in line else ""
                    # Only process folder names that do not contain "_", e.g., "Rn"
                    if "_" in folder_info:  
                        print(f"Skipping folder: {folder_info} as it contains '_'")
                        continue  # Skip this line

                    # If it doesn't contain "_", continue processing the folder
                    in_folder_section = True
                    other_rs.update(folder_info.split("_"))
                    continue

                if in_folder_section:
                    if line.startswith("Folder:"):
                        break  # Exit the section if a new folder starts
                    if line.strip() == "":
                        continue  # Skip empty lines
                    
                    if "No matching description found" in line:
                        continue
                    
                    clean_line = re.sub(r".*?\.smiles:", "", line).strip() 
                    if "forms" in clean_line:
                        ring_descriptions.add(clean_line.replace("forms ", ""))
                    elif clean_line:
                        descriptions.add(clean_line)

            sorted_descriptions = sorted(descriptions)
            if ring_descriptions:
                sorted_forms = sorted(ring_descriptions)

                # Step 1: Split "forms" description into parts and store as a list of tuples (main_part, r_group)
                forms_parts = []
                for form in sorted_forms:
                    parts = form.split(" with ")
                    if len(parts) == 2:
                        main_part, r_group = parts
                        forms_parts.append((main_part.strip(), r_group.strip()))
                    else:
                        forms_parts.append((parts[0].strip(), None))

                grouped_forms = {}
                for main_part, r_group in forms_parts:
                    if r_group:
                        if r_group not in grouped_forms:
                            grouped_forms[r_group] = []
                        grouped_forms[r_group].append(main_part)
                    else:
                        grouped_forms[None] = grouped_forms.get(None, []) + [main_part]

                final_forms = []
                forms_found = False
                collected_form_strs = []  # Store all form_str that matches "unsubstituted or substituted ring"

                # Process the groups
                for r_group, items in grouped_forms.items():
                    if r_group:
                        items_sorted = sorted(set(items))  
                        if len(items_sorted) > 1:
                            form_str = f"an unsubstituted/substituted {', '.join(items_sorted[:-1])} or {items_sorted[-1]} with {r_group}" 
                        else:
                            form_str = f"an unsubstituted/substituted {items_sorted[0]} with {r_group}"
                    else:
                        items_sorted = sorted(set(items)) 
                        form_str = f"an unsubstituted/substituted {', '.join(items_sorted[:-1])} or {items_sorted[-1]}" 

                    if "other similar ring" in form_str:
                        collected_form_strs.append((items_sorted, r_group))
                    else:
                        final_forms.append(form_str)

                if collected_form_strs:
                    items_sorted_all = []
                    r_group_all = [] 

                    # Collect and merge items_sorted and r_group
                    for items_sorted, r_group in collected_form_strs:
                        items_sorted_all.extend(items_sorted)
                        r_group_all.append(r_group)

                    # Remove "other similar ring" from items_sorted_all
                    items_sorted_all = [item for item in items_sorted_all if item != "other similar ring"]

                    # If items_sorted_all is empty, choose a fallback item randomly
                    if not items_sorted_all:
                        items_sorted_all = [random_choice]  

                    # Sort items_sorted and r_group
                    items_sorted_all = sorted(set(items_sorted_all))
                    r_group_all = sorted(set(r_group_all))

                    # If only one item, no need for commas or "or"/"and"
                    if len(items_sorted_all) > 1:
                        items_sorted_str = f"{', '.join(items_sorted_all[:-1])} or {items_sorted_all[-1]}"
                    else:
                        items_sorted_str = f"or {items_sorted_all}"

                    if len(r_group_all) > 1:
                        r_group_str = f"{', '.join(r_group_all[:-1])} and {r_group_all[-1]}"
                    else:
                        r_group_str = r_group_all[0]

                    # Merge items_sorted with "or" and r_group with "and"
                    forms_description = f"{items_sorted_str} with {r_group_str};"
                else:
                    forms_description = ""  #
                forms_description = forms_description.replace("['", "").replace("']", "") 

                if final_forms:
                    # Apply similar formatting for combined description
                    ring_description = "forms " + ", ".join(final_forms)
                    other_rs.discard(part)
                    if other_rs:
                        ring_description += f" with {' and '.join(other_rs)}"

                    sorted_descriptions.append(ring_description)

            if sorted_descriptions:
                if len(sorted_descriptions) > 1:
                    combined_description = ", ".join(sorted_descriptions[:-1]) + f" or {sorted_descriptions[-1]}"
                else:
                    combined_description = sorted_descriptions[0]
            else:
                print("No descriptions found.")

    else:
        print(f"Rule description file does not exist: {rule_description_path}")
        combined_description = ""  # If the file doesn't exist, set description to empty string

    if combined_description.strip().lower() == "hydrogen":
        combined_description = "C1-C4 straight or branched chain alkyl or hydrogen"
    
    return combined_description, forms_description


def output_images_from_r_folder(c, r_folder_path, y_position):
    """Directly output images from the current R folder and arrange them in four columns"""
    images = [f for f in os.listdir(r_folder_path) if f.lower().endswith(".png")]
    image_y_position = y_position - 2 * LINE_HEIGHT

    for row_start_index in range(0, len(images), 4):  # Process one row at a time (four images)
        row_images = images[row_start_index: row_start_index + 4]
        row_scaled_heights = []

        # Calculate the scaled dimensions and maximum height of all images in the current row
        for image_file in row_images:
            image_path = os.path.join(r_folder_path, image_file)
            img = Image.open(image_path)
            img = img.crop(img.getbbox())  # Remove surrounding whitespace from the image

            # Calculate scaled dimensions
            img_width, img_height = img.size
            if img_width >= img_height:
                scaled_width = 80  
                scaled_height = img_height * (scaled_width / img_width)
            else:
                scaled_height = 80  
                scaled_width = img_width * (scaled_height / img_height)

            row_scaled_heights.append(scaled_height)  # Save the height of each image

        max_scaled_height = max(row_scaled_heights)  # Maximum height of the images in the current row

        # Insert the images of the current row
        for index, image_file in enumerate(row_images):
            image_path = os.path.join(r_folder_path, image_file)
            img = Image.open(image_path)
            img = img.crop(img.getbbox())
            img_width, img_height = img.size
            if img_width >= img_height:
                scaled_width = 80  
                scaled_height = img_height * (scaled_width / img_width)
            else:
                scaled_height = 80  
                scaled_width = img_width * (scaled_height / img_height)

            # Prepare to insert the image
            bw_img = ImageReader(img)

            # Check if the page can accommodate four images
            if image_y_position - (LINE_HEIGHT + max_scaled_height + 5) < PAGE_BOTTOM:
                c.showPage()
                image_y_position = MARGIN_TOP  
                c.setFont("Times-Roman", 12) 

            # Set the X position for each image, ensuring four images per row
            if index == 0:  # First image
                x_position = MARGIN_LEFT + 10
            elif index == 1:  # Second image
                x_position = PAGE_WIDTH / 4 + MARGIN_LEFT - scaled_width / 2
            elif index == 2:  # Third image
                x_position = PAGE_WIDTH / 2 + MARGIN_LEFT - scaled_width / 2 - 40
            else:  # Fourth image
                x_position = 3 * PAGE_WIDTH / 4 - scaled_width / 2 

            # Insert the image
            c.drawImage(bw_img, x_position, image_y_position - max_scaled_height + 10, width=scaled_width, height=scaled_height)

            # Insert the English character
            text_x_position = x_position + scaled_width  
            text_y_position = image_y_position + 30 - max_scaled_height  
            if row_start_index + index == len(images) - 2:
                c.drawString(text_x_position, text_y_position, "or")
            else:
                c.drawString(text_x_position, text_y_position, ";")

        image_y_position -= (max_scaled_height)  

    return image_y_position


def generate_description():
    content_options = [
        "Cyclohexane, Cyclopentane, Benzene, Toluene",
        "4- to 7-membered ring, Benzene",
        "aromatic acyl which may have one halogen atom or an aryl as substituted",
        "C3-C6 cycloalkyl, -OH, -O(C1-C4 alkoxy)",
        "5- to 7-membered ring, optionally containing an O or NH"
    ]

    return random.choice(content_options)


def process_r_folder_recursively(c, r_folder, r_folder_path, y_position, index=0):  
    if "_" in r_folder:
        print(f"Skipping folder: {r_folder} (contains '_')")
        return y_position

    # Initialize full_output
    full_output = f"wherein {r_folder} is " if index == 0 else f"{r_folder} is "
    contains_nest_file_in_r_folder = any("nest" in file_name for file_name in os.listdir(r_folder_path))
    
    # If there are no nest files, directly output the description and stop further processing
    combined_description, forms_description = process_rule_description(r_folder, r_folder_path)
    if not contains_nest_file_in_r_folder:
        # If no nest files, directly output the description
        full_output += combined_description + ";"
        y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
        return y_position  # Stop further processing when there are no nest files

    # Handle the logic for nested folders
    y_position, nest_folder_path, success, R_nest_description, image_paths, form_paths = process_nest_folder(c, r_folder, r_folder_path, y_position)
    
    # Generate the description content
    full_output += combined_description
    if combined_description == "":
        full_output += f"{generate_description()}"

    if image_paths:
        full_output += f", {R_nest_description}"
        full_output = full_output.replace(";", ", ")
        full_output = full_output.replace("is ,", "")
        full_output = process_full_output_text(full_output)
        y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)                         
        if form_paths:
            y_position = generate_images_2(c, image_paths, y_position)
            full_output = "or forms the structure shown below:"
            y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
            y_position = generate_form_images(c, form_paths, y_position)
            full_output = forms_description
            y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
        else:
            y_position = generate_images(c, image_paths, y_position)
    else:
        full_output += f", {R_nest_description}"
        full_output = full_output.replace("is ,", "")
        if form_paths:
            full_output = full_output.replace(";", ",").replace(" or", ",")
            full_output += f"or forms the structure shown below:"
            full_output = full_output.replace(", or", " or")
            y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
            y_position = generate_form_images(c, form_paths, y_position)
            full_output = forms_description
            y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
        else:
            y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
    
    if not nest_folder_path or not os.path.isdir(nest_folder_path):
        return y_position

    nested_files = os.listdir(nest_folder_path)
    nested_r_folders = [
        f for f in nested_files if f.startswith("R") and os.path.isdir(os.path.join(nest_folder_path, f))
    ]
    
    processed_folders = {}
    for folder in nested_r_folders:
        r_parts = folder.split('_')
        for part in r_parts:
            number = int(re.search(r'\d+', part).group())
            if number not in processed_folders:
                processed_folders[number] = part

    for i, (number, nested_r_folder) in enumerate(sorted(processed_folders.items())):
        nested_r_folder_path = os.path.join(nest_folder_path, nested_r_folder)

        if os.path.isdir(nested_r_folder_path):
            y_position = process_r_folder_recursively(c, nested_r_folder, nested_r_folder_path, y_position, index=i)
    
    return y_position



def process_full_output_text(full_output):
    """Format the delimiters in the full_output content."""
    # If full_output contains "Rn is" (where n is any number), do not add "or"
    if re.search(r'\bR\d+\s+is\b', full_output):
        # If "Rn is" is found, replace all ";" and " or" with ","
        return full_output.replace(";", ",").replace(" or", ",")
    full_output = full_output.replace(";", ",").replace(" or", ",")
    
    # Ensure "or" is only added when there are multiple items
    if "," in full_output:
        parts = full_output.rsplit(",", 1)  # Split once from the right
        full_output = f"{parts[0]} or {parts[1]}"
    
    return full_output


def find_matching_nest_folder(base_folder_path, part_prefix):
    """
    Searches for folders in `base_folder_path` that contain `part_prefix` and 'nest' in any order.
    Matches folders like `R2_R3_R1_nest`, `R3_R1_nest`, etc., as long as they include the `part_prefix` and 'nest'.
    Returns the path to the nest folder if found, otherwise None.
    """
    try:
        # List subdirectories in the base folder
        subfolders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]

        # Define a regex pattern to match folders that contain the part_prefix and end in '_nest'
        # This pattern checks that the folder name includes part_prefix anywhere before '_nest'
        pattern = re.compile(rf".*{part_prefix}.*_nest$", re.IGNORECASE)

        # Search for the first folder that matches the pattern
        for folder in subfolders:
            if pattern.match(folder):
                return os.path.join(base_folder_path, folder)
    except FileNotFoundError:
        print(f"[Error] The base folder '{base_folder_path}' does not exist.")
    return None

def add_folder_description(c, part, folder_path, y_position, is_last_folder):
    """Add the folder description to the canvas at the specified y_position."""
    combined_description = ""
    forms_description = ""

    # Initialize a list to store X or Z folder descriptions temporarily
    if not hasattr(add_folder_description, "all_folder_descriptions"):
        add_folder_description.all_folder_descriptions = []

    if "R" in part:
        molecule_id_parts = part.split('_')  
        for i in range(len(molecule_id_parts)):
            current_part = '_'.join(molecule_id_parts[:i+1]) 
            
            # Check if "nest" exists in the folder
            contains_nest_file = any("nest" in file_name for file_name in os.listdir(folder_path))

            combined_description, forms_description = process_rule_description(current_part, folder_path)
            full_output = f"{current_part} is {combined_description}"
            if combined_description == "":
                full_output += f"{generate_description()}"
                       
            if not contains_nest_file:
                if forms_description:  
                    full_output += f" or forms {forms_description}" 
                else:
                    full_output += f"; "  
                y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)

            nest_folder_path = find_matching_nest_folder(folder_path, current_part)
            if nest_folder_path is None:
                print(f"[Warning] No appropriate 'nest' folder found for part '{current_part}' in folder '{folder_path}'")
            else:
                full_output += f", "  
                y_position, nest_folder_path, success, R_nest_description, image_paths, form_paths = process_nest_folder(
                    c, current_part, folder_path, y_position
                )              

                if image_paths:
                    full_output += R_nest_description
                    full_output = full_output.replace(";", ", ")
                    full_output = full_output.replace("is ,", "")
                    full_output = process_full_output_text(full_output)
                    y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)                                    
                    if form_paths:
                        y_position = generate_images_2(c, image_paths, y_position)
                        full_output = "or forms the structure shown below:"
                        y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
                        y_position = generate_form_images(c, form_paths, y_position)
                        full_output = forms_description
                        y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
                    else:
                        y_position = generate_images(c, image_paths, y_position)
                else:
                    full_output += R_nest_description
                    full_output = full_output.replace("is ,", "")
                    if form_paths:
                        full_output = full_output.replace(";", ",").replace(" or", ",")
                        full_output += f"or forms the structure shown below:"
                        full_output = full_output.replace(", or", " or")
                        y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
                        y_position = generate_form_images(c, form_paths, y_position)
                        full_output = forms_description
                        y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
                    else:
                        y_position = output_and_wrap(c, full_output, MARGIN_LEFT, y_position)
                                    
                
                if os.path.isdir(nest_folder_path):
                    try:
                        nested_files = os.listdir(nest_folder_path)
                        r_folders = sorted(
                            [f for f in nested_files if f.startswith("R") and os.path.isdir(os.path.join(nest_folder_path, f))],
                            key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf')
                        )
                        for index, r_folder in enumerate(r_folders):
                            r_folder_path = os.path.join(nest_folder_path, r_folder)
                            y_position = process_r_folder_recursively(c, r_folder, r_folder_path, y_position, index)
                    except FileNotFoundError:
                        print(f"[Error] The nest folder '{nest_folder_path}' could not be found.")

    if "X" in part or "Z" in part:
        replacements_path = os.path.join(folder_path, "replacements.txt")
        if os.path.exists(replacements_path):
            with open(replacements_path, "r") as file:
                replacement_lines = [line.replace("*", "").strip() for line in file if line.strip()]
                replacement_lines = list(set(replacement_lines))
                replacement_lines = [line for line in replacement_lines if line]

                common_elements = {"C", "H", "O", "N", "S"}
                halogens = {"F", "Cl", "Br", "I"}
                special_atoms = [line for line in replacement_lines if line not in common_elements and line not in halogens]

                if special_atoms:
                    combined_description = ", ".join(special_atoms)
                else:
                    if "N" in replacement_lines and len(replacement_lines) == 1:
                        combined_description = "C or N"
                    else:
                        has_common_elements = any(element in replacement_lines for element in common_elements)
                        has_halogens = any(element in replacement_lines for element in halogens)

                        combined_description = ""
                        if has_common_elements:
                            combined_description += "H, C, N, S or O" if "X" in part else "C, N, S or O"
                        elif has_halogens:
                            combined_description += "a halogen atom"
                        elif len(replacement_lines) > 1:
                            combined_description += ", ".join(replacement_lines[:-1]) + f" or {replacement_lines[-1]}"
                        elif replacement_lines:
                            combined_description += replacement_lines[0]

                if combined_description:
                    add_folder_description.all_folder_descriptions.append((part, combined_description))

    if is_last_folder and hasattr(add_folder_description, "all_folder_descriptions"):
        folder_descriptions = add_folder_description.all_folder_descriptions
        grouped_folders = {}

        for part, description in folder_descriptions:
            grouped_folders.setdefault(description, []).append(part)

        for description, parts in grouped_folders.items():
            if len(parts) > 1:
                part_list = ", ".join(parts[:-1]) + " and " + parts[-1]
                final_description = f"{part_list} are the same or different and each independently represents {description}"
            else:
                final_description = f"{parts[0]} represents {description}"

            final_description += ";" if description != list(grouped_folders.keys())[-1] else "."
            y_position = output_and_wrap(c, final_description, MARGIN_LEFT, y_position)

        add_folder_description.all_folder_descriptions = []

    return y_position


def output_and_wrap(c, text, start_x, y_position):
    """Output text on the canvas and wrap if it exceeds the page width."""
    text = text.replace("is forms", "forms")
    words = text.split(' ')
    current_line = ""

    PAGE_WIDTH = 600
    effective_width = PAGE_WIDTH - (MARGIN_LEFT + MARGIN_RIGHT)

    for word in words:
        new_line = current_line + (word if not current_line else " " + word)

        if c.stringWidth(new_line) > effective_width:
            if current_line:
                c.drawString(MARGIN_LEFT, y_position, current_line)
                y_position -= LINE_HEIGHT  
                current_line = word  
            else:
                current_line += word  
        else:
            current_line += (word if not current_line else " " + word)

        if y_position < PAGE_BOTTOM:
            c.showPage()  
            y_position = MARGIN_TOP 
            c.setFont("Times-Roman", 12)

    if current_line:
        c.drawString(MARGIN_LEFT, y_position, current_line)
        y_position -= LINE_HEIGHT

    return y_position

def create_pdf_in_current_directory(output_folder):
    pdf_path = os.path.join(output_folder, 'Claims.pdf')
    create_empty_pdf(pdf_path, output_folder)
    
