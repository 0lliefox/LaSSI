import csv
import sys
import yaml
import os
from collections import defaultdict


def reference_dates(csv_path):
    """Each case's reference date: the date part of its evidence's `date_time` (the notice's publication date, as
    written), which relative time expressions in the case are resolved against (LaSSI issue 158)."""
    dates = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            item_id = row['item_id']
            if item_id and row.get('date_time'):
                dates.setdefault(item_id[:-2], row['date_time'].strip()[:10])
    return dates


def write_reference_dates(csv_path, out_dir):
    """Write `<case>.meta.yaml` (the sidecar LaSSI reads `reference_date` from) for every case in the CSV."""
    for base_id, day in sorted(reference_dates(csv_path).items()):
        with open(os.path.join(out_dir, f"{base_id}.meta.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump({'reference_date': day}, f, default_flow_style=False)


def main():
    csv_path = 'neet/neet_v1.csv'
    out_dir = 'neet/evidence_cases'
    
    os.makedirs(out_dir, exist_ok=True)
    if '--meta-only' in sys.argv:
        # the case files have been edited by hand since they were generated; only (re)write the sidecars
        write_reference_dates(csv_path, out_dir)
        return
    
    cases = defaultdict(dict)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row['item_id']
            if not item_id:
                continue
            
            # e.g. transport_001_S -> base_id = transport_001, suffix = S
            base_id = item_id[:-2]
            suffix = item_id[-1]
            
            if 'evidence' not in cases[base_id]:
                cases[base_id]['evidence'] = row['evidence_text']
                
            if suffix == 'S':
                cases[base_id]['supported'] = row['claim_text']
            elif suffix == 'R':
                cases[base_id]['refuted'] = row['claim_text']
            elif suffix == 'N':
                cases[base_id]['not_enough_evidence'] = row['claim_text']

    for base_id, data in cases.items():
        yaml_path = os.path.join(out_dir, f"{base_id}.yaml")
        
        # Structure as requested: 4 distinct sentences
        yaml_data = [
            data.get('evidence', ''),
            data.get('supported', ''),
            data.get('refuted', ''),
            data.get('not_enough_evidence', '')
        ]
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False, width=float("inf"))
    write_reference_dates(csv_path, out_dir)

if __name__ == '__main__':
    main()
