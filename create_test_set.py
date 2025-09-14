import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# --- STEP 1: CONFIGURE YOUR PATHS ---
DATASET_BASE_DIR = 'ham10000_data'
OUTPUT_TEST_DIR = 'my_test_set'
# --- NEW: Define the output file for the labels ---
OUTPUT_LABELS_FILE = 'my_test_set_labels.csv'


# --- STEP 2: THE SCRIPT LOGIC (No changes needed in this part) ---
print("Starting script...")
METADATA_FILE = os.path.join(DATASET_BASE_DIR, 'HAM10000_metadata.csv')
IMG_DIR_1 = os.path.join(DATASET_BASE_DIR, 'HAM10000_images_part_1')
IMG_DIR_2 = os.path.join(DATASET_BASE_DIR, 'HAM10000_images_part_2')

try:
    df = pd.read_csv(METADATA_FILE)
    print(f"Successfully loaded metadata for {len(df)} images.")
except FileNotFoundError:
    print(f"ERROR: Metadata file not found at '{METADATA_FILE}'")
    exit()

image_paths = {os.path.splitext(f)[0]: os.path.join(p, f)
               for p in [IMG_DIR_1, IMG_DIR_2]
               for f in os.listdir(p)}
df['path'] = df['image_id'].map(image_paths.get)
df['label'] = df['dx']

_, test_df = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df['label']
)
print(f"Identified {len(test_df)} images for the test set.")

# --- Copy image files (same as before) ---
print(f"Creating test folder at: '{os.path.abspath(OUTPUT_TEST_DIR)}'")
os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)
copied_count = 0
for index, row in test_df.iterrows():
    source_path = row['path']
    if os.path.exists(source_path):
        shutil.copy(source_path, OUTPUT_TEST_DIR)
        copied_count += 1
print(f"Copied {copied_count} images to the '{OUTPUT_TEST_DIR}' folder.")


# --- STEP 3: NEW - SAVE THE LABELS FOR THE TEST SET ---
print(f"\nCreating labels file: '{OUTPUT_LABELS_FILE}'")

# Select only the image ID and the diagnosis code for our test set
test_labels_df = test_df[['image_id', 'dx']].copy()

# Optional: Add the full class name for readability
class_name_map = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}
test_labels_df['class_name'] = test_labels_df['dx'].map(class_name_map)

# Save this smaller DataFrame to a new CSV file
test_labels_df.to_csv(OUTPUT_LABELS_FILE, index=False)

print(f"\nDone! A file named '{OUTPUT_LABELS_FILE}' has been created with the actual labels.")
