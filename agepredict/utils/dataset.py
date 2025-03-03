from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AgesDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        df = pd.read_csv(csv_file)
        
        def extract_path(raw):
            if isinstance(raw, str) and raw.startswith('[') and raw.endswith(']'):
                trimmed = raw.strip('[]')
                parts = trimmed.split(',')
                return parts[0].strip().strip("'").strip('"')
            return raw
        
        valid_rows = []
        skipped = 0
        
        for i, row in df.iterrows():
            file_name = extract_path(row['full_path'])
            full_path = self.img_dir / file_name
            if full_path.exists():
                row['full_path'] = file_name 
                valid_rows.append(row)
            else:
                skipped += 1

        print(f"Found {len(valid_rows)} valid rows  Skipped {skipped} rows due to missing files.")
        self.data_frame = pd.DataFrame(valid_rows).reset_index(drop=True)

# stopped using this since it was not needed and perhbas causing too much filtering.

# def parse_face_location(loc_str):
#     # face locations look like [[111.29,111.29,252.66,252.66]] at times
#     if not isinstance(loc_str, str):
#         return None
    
#     # Converting the into comma separated string
#     fixed = loc_str.replace(" ", ",").replace(",,", ",")
    
#     try:
#         parsed = ast.literal_eval(fixed)
#         if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
#             # this parsed list should contain a list of 4 floats
#             # could be this for example [111.29,111.29,252.66,252.66]
#             coords = parsed[0] 
#             if len(coords) == 4:
#                 # this is the final tuple with the coodinate style (x1, y1, x2, y2)
#                 return tuple(coords) 
#     except:
#         pass
#     return None

# def compute_area(x1, y1, x2, y2):
#     width = x2 - x1
#     height = y2 - y1
#     return max(0, width) * max(0, height)


def balance_by_gender(df, factor=1.8):
    df_f = df[df['gender'] == 0.0].copy()
    df_m = df[df['gender'] == 1.0].copy()
    n_f = len(df_f)
    n_m = len(df_m)
    
    if n_f < n_m:
        target = int(n_f * factor) 
        target = min(target, n_m)
        df_f_oversampled = df_f.sample(n=target, replace=True)
        df_balanced = pd.concat([df_f_oversampled, df_m], ignore_index=True)
    else:
        target = int(n_m * factor)
        target = min(target, n_f)
        df_m_oversampled = df_m.sample(n=target, replace=True)
        df_balanced = pd.concat([df_f, df_m_oversampled], ignore_index=True)

    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced

def clean_dataset(csv_path, output_csv_path=None, min_age=1, max_age=100, min_face_score=1.0, do_balance_gender=True):

    print(f"CSV PATH!: {csv_path}")
    df = pd.read_csv(csv_path)
    original_len = len(df)

    df['face_score'] = df['face_score'].replace('#NAME?', np.nan)
    df['face_score'] = pd.to_numeric(df['face_score'], errors='coerce')

    mask_age = (df['age'] >= min_age) & (df['age'] <= max_age)
    mask_face = (df['face_score'].notna()) & (df['face_score'] >= min_face_score)

    df = df[mask_age & mask_face].copy()

    if do_balance_gender:
        df = balance_by_gender(df)

    df.reset_index(drop=True, inplace=True)
    
    cleaned_len = len(df)
    print(f"Original dataset size: {original_len}")
    print(f"Cleaned dataset size: {cleaned_len}")
    print(f"Removed {original_len - cleaned_len} rows.")

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
        print(f"Cleaned CSV saved to {output_csv_path}")

    return df

if __name__ == "__main__":
    root = Path("C:/Users/ramos/Desktop/GitHub/Kaggle-Competition-Age-Prediction/age-prediction-spring-25-at-cu-denver")
    
    original_csv = root / "wiki_labels.csv"
    cleaned_csv = root / "wiki_labels_clean.csv"

    cleaned_df = clean_dataset(
        csv_path=original_csv,
        output_csv_path=cleaned_csv,
        min_age=1,
        max_age=100,
        min_face_score=1.0,
        do_balance_gender=True
    )
    print("Cleaning complete.")