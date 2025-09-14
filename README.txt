UCCD3074: Deep Learning for Data Science Group Assignment
June Trimester 2025

Title: Deep Learning-Based Classification of Skin Diseases and Cancer

Application-Based

Group 5:
1. Loh Chia Heung (Leader) - 2301684
2. Tan Yi Xin - 2101990
3. Bester Loo Man Ting - 2207066
4. Cornelius Wong Qin Jun - 2104603

Submitted File:
1. Report (PDF)
- Assignment2_group5_LohChiaHeung.pdf

2. Source Code
- Assignment2_group5_LohChiaHeung.ipynb

########    Instructions:   ########

- The code is designed to run automatically on Kaggle.
- Upload the Assignment2_Group5_LohChiaHeung.ipynb to Kaggle.
- Select GPU accelerator (GPU T4 x2 in our case)
- Click "Run All" to execute the code. (It will take around 2-3 hours to complete)

########    End of Instructions   ########

Best Model: Swin Transformer

3. Demo Application
- main.py
- create_test_set.py
- requirements.txt

########    Instructions:   ########

Step 1: Download Model
    - Download the fine-tuned best model (swin_transformer.pth) from the Model Path Link in Section 4.

Step 2: Download Dataset
    - Download the HAM10000 dataset from Kaggle link in Section 4.
    ham10000_data/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    └── HAM10000_images_part_2/

Step 3: Run the following command: python create_test_set.py
This script creates:
    my_test_set/ → a folder containing ~1000 test images
    my_test_set_labels.csv → a CSV file with the ground truth labels

Step 4: Launch the Dermatology AI Assistant Application with:
    python main.py

Step 5: Using the GUI
    Load Model
    1. Click Load Model and select your downloaded .pth file (e.g., swin_transformer.pth).
        - Once loaded, the model status indicator will turn green.
    2. Analyze Image(s)
        a. Single Image
            - Click Select Single Image and choose an image file, or
            - Drag & drop an image directly onto the Image Preview area.
        b. Batch of Images
            - Click Select Folder for Batch and select the my_test_set/ folder created in Step 2.
            - The app will process all images and display predictions, ground truth, and correctness.
    3. Export Results
        - After a batch run, go to File > Export Batch Results to CSV…
        - Save the prediction results for further analysis.

########    End of Instructions   ########

4. ReadMe
- README.txt

Kaggle Dataset Link:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Google Drive Link (Model Path):
https://drive.google.com/drive/u/1/folders/1KFW1Jy0uz-RkW2A7wW6i0B8KVGKAfTSQ

Install dependencies with:
pip install -r requirements.txt

GitHub Link:
https://github.com/LohChiaHeung/UCCD3074_Group5







