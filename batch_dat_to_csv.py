from pathlib import Path
import pandas as pd
import numpy as np
import sys
from datetime import datetime

FIELD_POSITIONS = [
    ("year_quarter", 1, 5, 5, None),
    ("household_ref_no", 6, 16, 11, '99999999999'),
    ("relationship", 17, 18, 2, '99'),
    ("age", 19, 20, 2, '99'),
    ("sex", 23, 23, 1, '9'),
    ("education", 24, 24, 1, '9'),
    ("marital_status", 25, 25, 1, '9'),
    ("economic_status", 26, 26, 1, '9'),
    ("underemployed", 27, 27, 1, '9'),
    ("work_place", 28, 28, 1, '9'),
    ("industry_employed", 29, 30, 2, '99'),
    ("industry_underemployed", 31, 31, 1, '9'),
    ("previous_industry", 32, 32, 1, '9'),
    ("occupation_employed", 33, 34, 2, '99'),
    ("previous_occupation", 35, 36, 2, '99'),
    ("hours_7days", 37, 39, 3, '999'),  # 999 is NA, but 99 is valid ("99 or over")
    ("hours_main_employment", 40, 42, 3, '999'),  # 999 is NA, but 99 is valid
    ("earnings_employed", 43, 44, 2, '99'),
    ("earnings_underemployed", 45, 46, 2, '99'),
    ("unemployment_duration", 47, 47, 1, '9'),
    ("leave_reason", 48, 48, 1, '9'),
    ("foreign_helper", 49, 49, 1, '9'),
    ("grossing_up_factor", 50, 60, 11, None),  # Keep grossing-up factor
    ("spouse_serial", 61, 62, 2, '99'),
    ("parent_serial", 63, 64, 2, '99'),
    ("year_completed_study", 65, 68, 4, '9999'),
    ("length_employment", 69, 70, 2, '99'),
    ("intend_same_industry", 71, 71, 1, '9'),
    ("intend_industry", 72, 72, 1, '9'),
    ("intend_same_occupation", 73, 73, 1, '9'),
    ("intend_occupation", 74, 75, 2, '99'),
    ("person_serial", 76, 77, 2, None),  # No NA for person serial number
    ("hours_secondary", 78, 80, 3, '999'),  # 999 is NA, but 99 is valid
    ("earnings_main", 81, 82, 2, '99'),
    ("earnings_other", 83, 84, 2, '99'),
    ("other_employment_7days", 85, 85, 1, '9'),
    ("cny_bonus", 86, 87, 2, '99'),
]

# === PARSING FUNCTIONS ===

def parse_line(line, field_positions):
    """Parse a single line of fixed-width data using position specifications"""
    fields = {}
    for field_name, start_pos, end_pos, field_width, na_code in field_positions:
        # Convert 1-indexed positions to 0-indexed for Python
        value = line[start_pos-1:end_pos].strip()
        
        # Check if value matches the field-specific NA code
        if na_code and value == na_code:
            value = ''
        
        fields[field_name] = value
    return fields

def add_calculated_columns(df):
    """Add calculated columns based on DAX formulas"""
    
    # Convert necessary columns to numeric
    df['economic_status'] = pd.to_numeric(df['economic_status'], errors='coerce')
    df['hours_7days'] = pd.to_numeric(df['hours_7days'], errors='coerce')
    df['underemployed'] = pd.to_numeric(df['underemployed'], errors='coerce')
    df['earnings_employed'] = pd.to_numeric(df['earnings_employed'], errors='coerce')
    df['earnings_underemployed'] = pd.to_numeric(df['earnings_underemployed'], errors='coerce')
    
    # Is_Employed
    df['Is_Employed'] = ((df['economic_status'] >= 1) & (df['economic_status'] <= 4))
    
    # Weekly_Hours
    df['Weekly_Hours'] = df['hours_7days'].apply(
        lambda x: np.nan if pd.isna(x) or x == 999 else x
    )
    
    # Monthly_Hours_Approx
    df['Monthly_Hours_Approx'] = df['Weekly_Hours'].apply(
        lambda x: x * (52.0 / 12) if pd.notna(x) and x > 0 else np.nan
    )
    
    # Hourly_FullyEmployed
    def calc_hourly_fully_employed(row):
        if not row['Is_Employed']:
            return np.nan
        if row['underemployed'] == 1:
            return np.nan
        if pd.isna(row['earnings_employed']) or row['earnings_employed'] == 99:
            return np.nan
        
        # Map earnings_employed to midpoint
        earnings_map = {
            1: 1500, 2: 2500, 3: 3500, 4: 4500, 5: 5500,
            6: 6500, 7: 7500, 8: 8500, 9: 9500, 10: 10500,
            11: 11500, 12: 12500, 13: 13500, 14: 14500, 15: 15500,
            16: 16500, 17: 17500, 18: 18500, 19: 19500, 20: 21000,
            21: 23000, 22: 25000, 23: 27000, 24: 29000, 25: 31000,
            26: 33000, 27: 35000, 28: 37000, 29: 39000, 30: 41000,
            31: 43000, 32: 45000, 33: 47000, 34: 49000, 35: 55000,
            36: 65000, 37: 75000, 38: 85000, 39: 95000, 40: 120000
        }
        
        mid_earnings = earnings_map.get(row['earnings_employed'], np.nan)
        
        if pd.notna(mid_earnings) and pd.notna(row['Monthly_Hours_Approx']):
            return mid_earnings / row['Monthly_Hours_Approx']
        return np.nan
    
    df['Hourly_FullyEmployed'] = df.apply(calc_hourly_fully_employed, axis=1)
    
    # Hourly_Underemployed
    def calc_hourly_underemployed(row):
        if not row['Is_Employed']:
            return np.nan
        if row['underemployed'] != 1:
            return np.nan
        if pd.isna(row['earnings_underemployed']) or row['earnings_underemployed'] == 99:
            return np.nan
        
        # Map earnings_underemployed to midpoint
        earnings_map = {
            1: 1500, 2: 2500, 3: 3500, 4: 4500, 5: 5500,
            6: 6500, 7: 7500, 8: 8500, 9: 9500, 10: 15000
        }
        
        mid_earnings = earnings_map.get(row['earnings_underemployed'], np.nan)
        
        if pd.notna(mid_earnings) and pd.notna(row['Monthly_Hours_Approx']):
            return mid_earnings / row['Monthly_Hours_Approx']
        return np.nan
    
    df['Hourly_Underemployed'] = df.apply(calc_hourly_underemployed, axis=1)
    
    # Hourly_Combined
    df['Hourly_Combined'] = df['Hourly_FullyEmployed'].fillna(df['Hourly_Underemployed'])
    
    # Wage Bin Custom
    def get_wage_bin(hourly_wage):
        if pd.isna(hourly_wage):
            return "(No Wage Data)"
        elif hourly_wage < 10:
            return "<10"
        elif hourly_wage < 20:
            return "10-19.99"
        elif hourly_wage < 30:
            return "20-29.99"
        elif hourly_wage < 40:
            return "30-39.99"
        elif hourly_wage < 50:
            return "40-49.99"
        elif hourly_wage < 60:
            return "50-59.99"
        elif hourly_wage < 70:
            return "60-69.99"
        elif hourly_wage < 80:
            return "70-79.99"
        elif hourly_wage < 90:
            return "80-89.99"
        elif hourly_wage < 100:
            return "90-99.99"
        else:  # >= 100
            return "100+"
    
    df['Wage_Bin_Custom'] = df['Hourly_Combined'].apply(get_wage_bin)
    
    # Wage Bin Sort Order
    bin_sort_map = {
        "(No Wage Data)": 0,
        "<10": 1,
        "10-19.99": 2,
        "20-29.99": 3,
        "30-39.99": 4,
        "40-49.99": 5,
        "50-59.99": 6,
        "60-69.99": 7,
        "70-79.99": 8,
        "80-89.99": 9,
        "90-99.99": 9,  # Note: Your formula showed "80-99.99" in the bin list
        "100+": 10
    }
    df['Wage_Bin_Sort_Order'] = df['Wage_Bin_Custom'].map(bin_sort_map).fillna(99).astype(int)
    
    return df

def process_file(input_path):
    """Process a single .dat file and convert to CSV"""
    print(f"\nProcessing: {input_path.name}")
    
    if not input_path.exists():
        print(f"  [X] File not found")
        return False
    
    try:
        rows = []
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n\r')
                if not line:
                    continue
                try:
                    parsed = parse_line(line, FIELD_POSITIONS)
                    rows.append(parsed)
                except Exception as e:
                    print(f"  [WARNING] Error parsing line {line_num}: {e}")
                    continue
        
        if not rows:
            print(f"  [X] No valid data parsed")
            return False
            
        df = pd.DataFrame(rows)
        
        # Add calculated columns
        df = add_calculated_columns(df)
        
        # Save to CSV in the same directory
        output_path = input_path.with_suffix('.csv')
        df.to_csv(output_path, index=False)
        
        print(f"  [OK] Parsed {len(df)} records -> {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

# === MAIN ===

if __name__ == "__main__":
    print("=" * 70)
    print("Batch GHS Data Parser - Q1 2008 onwards format")
    print("=" * 70)
    
    # Define base directory
    base_dir = Path(__file__).parent / "Microdata_GHS"
    
    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)
    
    # Find all .dat files in PP folders (case-insensitive)
    print("\nSearching for .dat files in PP folders...")
    dat_files = []
    
    # Search in all subdirectories
    for path in base_dir.rglob("*.dat"):
        # Check if "pp" is in the path (case-insensitive)
        if any(part.lower() == "pp" for part in path.parts):
            dat_files.append(path)
    
    if not dat_files:
        print(f"No .dat files found in PP folders under {base_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(dat_files)} .dat file(s) in PP folders")
    print("=" * 70)
    
    # Process each file
    success_count = 0
    start_time = datetime.now()
    
    for dat_file in sorted(dat_files):
        if process_file(dat_file):
            success_count += 1
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total files: {len(dat_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(dat_files) - success_count}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print("=" * 70)
    
    if success_count > 0:
        print("\n[SUCCESS] Done! CSV files have been created next to each .dat file.")
    else:
        print("\n[FAILED] No files were successfully processed.")
