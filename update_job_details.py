import os
import json
import re

def process_job_details(file_path):
    print(f"Processing {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        return

    header = lines[0].strip()
    separator = lines[1].strip()
    rows = lines[2:]

    if "| geometry |" in header:
        print(f"Re-processing {file_path} to fill N/A values.")
        # We need to extract the existing data and rebuild
        new_header = header
        new_separator = separator
    else:
        new_header = header + " geometry |"
        new_separator = separator + " --- |"
    
    new_lines = [new_header + "\n", new_separator + "\n"]
    
    base_dir = os.path.dirname(file_path)
    
    for row in rows:
        row = row.strip()
        if not row:
            continue
        
        parts = [p.strip() for p in row.split('|')]
        # parts[0] is empty because row starts with |
        # parts[1] should be the time tag
        if len(parts) < 2:
            new_lines.append(row + "\n")
            continue
            
        time_tag = parts[1]
        
        # Check if we already have geometry (if it was already there and not N/A)
        existing_geometry = "N/A"
        if "| geometry |" in header and len(parts) > 5:
             # geometry is likely at index 5 (0:empty, 1:time_tag, 2:device, 3:shots, 4:uhf_spin_seed, 5:geometry)
             existing_geometry = parts[5]
        
        if existing_geometry != "N/A":
            new_lines.append(row + "\n")
            continue

        metadata_path = os.path.join(base_dir, time_tag, "metadata.json")
        
        geometry_str = "N/A"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    geometry = metadata.get("geometry", "N/A")
                    geometry_str = " ".join(geometry.split())
            except Exception as e:
                print(f"Error reading {metadata_path}: {e}")
        else:
            # Recursive search for metadata.json in base_dir
            found = False
            for root, dirs, files in os.walk(base_dir):
                if time_tag in root and "metadata.json" in files:
                    try:
                        with open(os.path.join(root, "metadata.json"), 'r') as f:
                            metadata = json.load(f)
                            geometry = metadata.get("geometry", "N/A")
                            geometry_str = " ".join(geometry.split())
                            found = True
                            break
                    except Exception as e:
                        print(f"Error reading {os.path.join(root, 'metadata.json')}: {e}")
            
            if not found:
                 # Check if time_tag is actually inside any subdirectory
                 # The search_project showed that 20260130-212326 is INSIDE 20260130-212325
                 for root, dirs, files in os.walk(base_dir):
                     if time_tag in dirs:
                         potential_metadata = os.path.join(root, time_tag, "metadata.json")
                         if os.path.exists(potential_metadata):
                             try:
                                 with open(potential_metadata, 'r') as f:
                                     metadata = json.load(f)
                                     geometry = metadata.get("geometry", "N/A")
                                     geometry_str = " ".join(geometry.split())
                                     found = True
                                     break
                             except:
                                 pass
                     if found: break

        if "| geometry |" in header:
            # Replace N/A with geometry_str in the last column
            # We assume geometry is the last column
            parts[5] = geometry_str
            new_row = " | ".join(parts).strip()
            # Restore leading/trailing pipes if join didn't
            if not new_row.startswith("|"): new_row = "| " + new_row
            if not new_row.endswith("|"): new_row = new_row + " |"
            new_lines.append(new_row + "\n")
        else:
            new_lines.append(f"{row} {geometry_str} |\n")

    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def main():
    root_dir = os.path.join("research", "14_aws_jobs", "data")
    
    # 1. Gather all metadata info first from the entire root_dir
    global_metadata_map = {}
    for r, d, f in os.walk(root_dir):
        if "metadata.json" in f:
            path = os.path.join(r, "metadata.json")
            try:
                with open(path, 'r') as meta_f:
                    meta = json.load(meta_f)
                    geo = meta.get("geometry", "N/A")
                    geo_str = " ".join(geo.split())
                    
                    # Store by its immediate directory name
                    dir_name = os.path.basename(r)
                    global_metadata_map[dir_name] = geo_str
                    
                    # Also check for any subdirectories that might have the time tag
                    for item in os.listdir(r):
                        if os.path.isdir(os.path.join(r, item)):
                            match = re.search(r'(\d{8}-\d{6})', item)
                            if match:
                                global_metadata_map[match.group(1)] = geo_str
            except:
                pass
    
    # 2. Process each job_details.md
    for root, dirs, files in os.walk(root_dir):
        if "job_details.md" in files:
            file_path = os.path.join(root, "job_details.md")
            print(f"Processing {file_path}")
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if not lines: continue

            header = lines[0].strip()
            separator = lines[1].strip()
            rows = lines[2:]

            if "| geometry |" not in header:
                new_header = header + " geometry |"
                new_separator = separator + " --- |"
            else:
                new_header = header
                new_separator = separator
            
            new_lines = [new_header + "\n", new_separator + "\n"]
            
            for row in rows:
                row = row.strip()
                if not row: continue
                parts = [p.strip() for p in row.split('|')]
                if len(parts) < 2:
                    new_lines.append(row + "\n")
                    continue
                
                time_tag = parts[1]
                geometry_str = global_metadata_map.get(time_tag, "N/A")
                
                if "| geometry |" in header:
                    if len(parts) > 5:
                        if parts[5] == "N/A" or not parts[5]:
                            parts[5] = geometry_str
                        # else keep existing
                    else:
                        parts.append(geometry_str)
                        parts.append("") # for trailing pipe if needed
                    
                    # Rebuild row
                    # parts[0] is empty, parts[-1] might be empty
                    new_row = " | ".join(parts).strip()
                    if not new_row.startswith("|"): new_row = "| " + new_row
                    if not new_row.endswith("|"): new_row = new_row + " |"
                    new_lines.append(new_row + "\n")
                else:
                    new_lines.append(f"{row} {geometry_str} |\n")

            with open(file_path, 'w') as f:
                f.writelines(new_lines)

if __name__ == "__main__":
    main()
