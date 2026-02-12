import csv
import json
import os
import glob

csv_path = 'research/14_aws_jobs/jobs_summary.csv'
data_root = 'research/14_aws_jobs/data'

def get_top_bitstring(timestamp):
    # Search for sim_statevector.json in any subdirectory under data_root that matches the timestamp
    pattern = os.path.join(data_root, '*', timestamp, 'sim_statevector.json')
    files = glob.glob(pattern)
    
    if not files:
        print(f"Warning: No sim_statevector.json found for timestamp {timestamp}")
        return None
    
    with open(files[0], 'r') as f:
        data = json.load(f)
        if not data or 'amplitudes' not in data:
            return None
        
        n_qubits = data.get('n_qubits', 0)
        max_prob = -1.0
        top_index = -1
        
        for amp in data['amplitudes']:
            prob = amp['real']**2 + amp['imag']**2
            if prob > max_prob:
                max_prob = prob
                top_index = amp['index']
        
        if top_index != -1:
            # Convert index to bitstring
            return format(top_index, f'0{n_qubits}b')
        return None

def update_csv():
    rows = []
    header = []
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            timestamp = row['timestamp']
            top_sim = get_top_bitstring(timestamp)
            row['top simulated bitstring'] = top_sim
            rows.append(row)
    
    if 'top simulated bitstring' not in header:
        header.append('top simulated bitstring')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    update_csv()
    print("Successfully updated jobs_summary.csv with 'top simulated bitstring' column.")
