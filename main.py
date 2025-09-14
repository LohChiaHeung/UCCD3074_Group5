# ############################ 
# UI adapted/generated with LLM 
# Features (model, data, evaluation) coded by Tan Yi Xin 
# ############################

# =============================================================================
# Dermatology AI Assistant - Full Application
#
# Description:
# A comprehensive GUI application for classifying skin lesion images using a
# trained PyTorch model. It supports single-image analysis, batch processing,
# and direct comparison with ground truth labels from the HAM10000 dataset.
#
# Features:
# - Load various model architectures (Swin, EfficientNet, DenseNet, Inception).
# - Predict on single images via file dialog or drag-and-drop.
# - Process entire folders of images in a separate thread to keep the UI responsive.
# - Displays a probability distribution chart for each prediction.
# - For batch processing, it automatically loads 'HAM10000_metadata.csv' to
#   retrieve the actual class for each image.
# - The results table shows Prediction vs. Actual and a Correct/Incorrect status.
# - Results can be exported to a CSV file.
# - Saves and loads window size and layout for a better user experience.
#
# =============================================================================


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import csv
import json
import threading
import queue
import numpy as np
from tkinterdnd2 import DND_FILES, TkinterDnD
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Configuration ---
CLASS_NAMES = [
    'Actinic keratoses (akiec)', 
    'Basal cell carcinoma (bcc)', 
    'Benign keratosis-like lesions (bkl)', 
    'Dermatofibroma (df)', 
    'Melanoma (mel)', 
    'Melanocytic nevi (nv)', 
    'Vascular lesions (vasc)'
]
MALIGNANT_CLASSES = {'Melanoma (mel)', 'Basal cell carcinoma (bcc)'}
SETTINGS_FILE = 'app_settings.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Loading Logic ---
def get_model_architecture(model_path, num_classes):
    model_name = os.path.basename(model_path).lower()
    if 'swin' in model_name:
        model, arch_name = models.swin_t(weights=None), "Swin Transformer"
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif 'efficientnet' in model_name:
        model, arch_name = models.efficientnet_b0(weights=None), "EfficientNet-B0"
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif 'densenet' in model_name:
        model, arch_name = models.densenet121(weights=None), "DenseNet-121"
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'inception' in model_name:
        model, arch_name = models.inception_v3(weights=None, aux_logits=False), "InceptionV3"
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture from filename: {model_name}. Filename must contain 'swin', 'efficientnet', 'densenet', or 'inception'.")
    return model, arch_name

# --- Helper Class for Tooltips ---
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief='solid', borderwidth=1, font=("Segoe UI", 8))
        label.pack()
    def hide(self, event=None):
        if self.tooltip: self.tooltip.destroy()
        self.tooltip = None

# --- Main Application Class ---
class SkinLesionClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.model, self.model_arch, self.input_size = None, "N/A", 224
        self.batch_folder_path = None
        self.result_queue = queue.Queue()
        self.ground_truth_map = {}

        self.CLASS_MAP = {
            'akiec': 'Actinic keratoses (akiec)', 'bcc': 'Basal cell carcinoma (bcc)',
            'bkl': 'Benign keratosis-like lesions (bkl)', 'df': 'Dermatofibroma (df)',
            'mel': 'Melanoma (mel)', 'nv': 'Melanocytic nevi (nv)', 'vasc': 'Vascular lesions (vasc)'
        }

        self._configure_styles()
        self._create_widgets()
        self._load_settings()

    def _configure_styles(self):
        self.BG_COLOR = '#f8f9fa'
        self.ACCENT_COLOR = '#4e73df'
        self.SECONDARY_COLOR = '#858796'
        self.SUCCESS_COLOR = '#1cc88a'
        self.DANGER_COLOR = '#e74a3b'
        self.TEXT_COLOR = '#5a5c69'
        self.LIGHT_BG = '#eaecf4'
        self.root.configure(bg=self.BG_COLOR)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=self.BG_COLOR, foreground=self.TEXT_COLOR, font=('Segoe UI', 10))
        style.configure('TFrame', background=self.BG_COLOR)
        style.configure('TLabel', background=self.BG_COLOR)
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground=self.ACCENT_COLOR)
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=8, borderwidth=0, relief=tk.FLAT)
        style.map('TButton', background=[('active', '#3652b1'), ('!disabled', self.ACCENT_COLOR)], foreground=[('!disabled', 'white')])
        style.configure('TLabelframe', background=self.BG_COLOR, bordercolor=self.LIGHT_BG, relief=tk.SOLID, borderwidth=1)
        style.configure('TLabelframe.Label', font=('Segoe UI', 11, 'bold'), foreground=self.ACCENT_COLOR, background=self.BG_COLOR, padding=(10, 5))
        style.configure('Treeview', background='white', foreground=self.TEXT_COLOR, fieldbackground='white', rowheight=28, font=('Segoe UI', 9))
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'), background=self.LIGHT_BG, foreground=self.TEXT_COLOR, relief=tk.FLAT)
        style.map('Treeview', background=[('selected', self.ACCENT_COLOR)], foreground=[('selected', 'white')])
        style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])

    def _create_widgets(self):
        self._create_menu()
        header = ttk.Frame(self.root, padding=(15, 10)); header.pack(fill=tk.X)
        ttk.Label(header, text="Dermatology AI Assistant", style='Title.TLabel').pack(side=tk.LEFT)
        status_frame = ttk.Frame(header); status_frame.pack(side=tk.RIGHT)
        self.status_indicator = tk.Canvas(status_frame, width=20, height=20, bg=self.BG_COLOR, highlightthickness=0); self.status_indicator.pack(side=tk.RIGHT, padx=(5, 0))
        self.status_indicator.create_oval(5, 5, 15, 15, fill=self.DANGER_COLOR, outline='', tags='status')
        ttk.Label(status_frame, text="Model Status:", font=('Segoe UI', 9)).pack(side=tk.RIGHT)
        
        main_frame = ttk.Frame(self.root, padding=(15, 5)); main_frame.pack(fill=tk.BOTH, expand=True)
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL); self.paned_window.pack(fill=tk.BOTH, expand=True)

        self._create_left_panel()
        self._create_right_panel()
        
        status_bar = ttk.Frame(self.root, padding=(15, 5)); status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready. Please load a model.")
        ttk.Label(status_bar, textvariable=self.status_var, font=('Segoe UI', 9)).pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(status_bar, mode='determinate', length=200); self.progress.pack(side=tk.RIGHT)

    def _create_menu(self):
        menubar = tk.Menu(self.root); self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Model...", command=self.load_model_dialog)
        file_menu.add_command(label="Export Batch Results to CSV...", command=self._export_to_csv)
        file_menu.add_separator(); file_menu.add_command(label="Exit", command=self._on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=0); help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _create_left_panel(self):
        left_frame = ttk.Frame(self.paned_window, padding=5); self.paned_window.add(left_frame, weight=3)
        image_frame = ttk.LabelFrame(left_frame, text="Image Preview", padding=10); image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.image_canvas = tk.Canvas(image_frame, bg='white', highlightthickness=0); self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.drop_hint_id = self.image_canvas.create_text(0, 0, text="Drag & Drop Image Here\nor Load a Model to Begin", font=('Segoe UI', 12, 'italic'), fill=self.SECONDARY_COLOR, tags="drop_hint")
        self.root.bind('<Configure>', lambda e: self._center_drop_hint())
        self.image_canvas.drop_target_register(DND_FILES); self.image_canvas.dnd_bind('<<Drop>>', self.handle_drop)
        results_frame = ttk.LabelFrame(left_frame, text="Batch Processing Results", padding=10); results_frame.pack(fill=tk.BOTH, expand=True)
        self._create_results_treeview(results_frame)

    def _create_right_panel(self):
        right_frame = ttk.Frame(self.paned_window, padding=5); self.paned_window.add(right_frame, weight=2)
        model_frame = ttk.LabelFrame(right_frame, text="1. Model Configuration", padding=10); model_frame.pack(fill=tk.X, pady=(0, 5), ipady=5)
        self.load_model_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model_dialog); self.load_model_btn.pack(pady=5, padx=10, fill=tk.X)
        self.model_info_label = ttk.Label(model_frame, text="Architecture: N/A\nInput Size: N/A", font=('Segoe UI', 9), justify=tk.LEFT); self.model_info_label.pack(pady=5, padx=10, fill=tk.X)
        Tooltip(self.load_model_btn, "Load a trained .pth model file.")
        image_select_frame = ttk.LabelFrame(right_frame, text="2. Image Selection", padding=10); image_select_frame.pack(fill=tk.X, pady=5, ipady=5)
        self.select_img_btn = ttk.Button(image_select_frame, text="Select Single Image", command=self.select_image_dialog, state=tk.DISABLED); self.select_img_btn.pack(pady=5, padx=10, fill=tk.X)
        self.select_folder_btn = ttk.Button(image_select_frame, text="Select Folder for Batch", command=self._start_batch_process, state=tk.DISABLED); self.select_folder_btn.pack(pady=(5, 10), padx=10, fill=tk.X)
        Tooltip(self.select_img_btn, "Analyze a single skin lesion image."); Tooltip(self.select_folder_btn, "Analyze all images in a selected folder.")
        result_frame = ttk.LabelFrame(right_frame, text="3. Prediction Distribution", padding=10); result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.fig = plt.figure(figsize=(4, 3), dpi=100); self.fig.patch.set_facecolor(self.BG_COLOR); self.ax = self.fig.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=result_frame); self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._update_probability_chart(np.zeros(len(CLASS_NAMES)))
        
        legend_frame = ttk.LabelFrame(right_frame, text="Result Color Key", padding=10); legend_frame.pack(fill=tk.X, pady=5)
        tk.Canvas(legend_frame, width=15, height=15, bg='#e8f5e9', highlightthickness=1, highlightbackground='#2e7d32').grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(legend_frame, text="Correct Prediction", font=('Segoe UI', 9)).grid(row=0, column=1, sticky=tk.W)
        tk.Canvas(legend_frame, width=15, height=15, bg='#ffebee', highlightthickness=1, highlightbackground='#c62828').grid(row=1, column=0, padx=5, pady=2)
        ttk.Label(legend_frame, text="Incorrect Prediction", font=('Segoe UI', 9)).grid(row=1, column=1, sticky=tk.W)

    def _create_results_treeview(self, parent):
        tree_frame = ttk.Frame(parent); tree_frame.pack(fill=tk.BOTH, expand=True)
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL); h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        
        self.tree = ttk.Treeview(tree_frame, columns=('File', 'Prediction', 'Actual', 'Correct?', 'Confidence'), show='headings', yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.config(command=self.tree.yview); h_scroll.config(command=self.tree.xview)
        
        cols = {'File': 150, 'Prediction': 180, 'Actual': 180, 'Correct?': 60, 'Confidence': 80}
        headings = {'Actual': 'Actual Class', 'Correct?': 'Correct?'}
        for col, width in cols.items():
            text = headings.get(col, col)
            anchor = tk.CENTER if col in ['Correct?', 'Confidence'] else tk.W
            self.tree.heading(col, text=text, command=lambda _c=col: self._sort_treeview_column(_c, False))
            self.tree.column(col, width=width, anchor=anchor)

        self.tree.tag_configure('correct', background='#e8f5e9', foreground='#2e7d32')
        self.tree.tag_configure('incorrect', background='#ffebee', foreground='#c62828')
        
        self.tree.grid(row=0, column=0, sticky='nsew'); v_scroll.grid(row=0, column=1, sticky='ns'); h_scroll.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1); tree_frame.grid_columnconfigure(0, weight=1)
        self.tree.bind('<Double-1>', self.on_tree_double_click)

    def _load_metadata(self, folder_path):
        """
        Finds and loads a metadata/labels file to create a ground truth map.
        It prioritizes finding a specific labels file (like 'my_test_set_labels.csv')
        before falling back to the original full metadata file.
        """
        self.ground_truth_map = {}
        folder_name = os.path.basename(folder_path) # e.g., "my_test_set"
        
        # --- NEW: Prioritize the specific labels file ---
        # e.g., looks for "my_test_set_labels.csv"
        specific_labels_file = f"{folder_name}_labels.csv"

        # List of potential file paths to check, in order of priority
        potential_paths = [
            os.path.join(os.path.dirname(folder_path), specific_labels_file), # e.g., MyProject/my_test_set_labels.csv
            os.path.join(folder_path, specific_labels_file),                  # e.g., MyProject/my_test_set/my_test_set_labels.csv
            # Fallback to the original full metadata file
            os.path.join(os.path.dirname(folder_path), 'HAM10000_metadata.csv'),
            os.path.join(folder_path, 'HAM10000_metadata.csv')
        ]
        
        metadata_path = next((path for path in potential_paths if os.path.exists(path)), None)

        if metadata_path:
            try:
                import pandas as pd
                metadata_df = pd.read_csv(metadata_path)
                
                # Check if the CSV has the 'dx' column (from HAM10000_metadata) or a 'class_name' column
                if 'dx' in metadata_df.columns:
                    # Create the map from the original file's 'dx' column
                    self.ground_truth_map = {row['image_id']: self.CLASS_MAP[row['dx']] for _, row in metadata_df.iterrows()}
                    self.status_var.set("Full metadata loaded. Ground truth will be displayed.")
                elif 'class_name' in metadata_df.columns:
                    # Create the map from the new labels file's 'class_name' column
                    # We need to find the full name from our CLASS_NAMES list
                    # This is slightly more complex to match "Melanoma" to "Melanoma (mel)"
                    name_lookup = {name.split(' (')[0]: name for name in CLASS_NAMES}
                    self.ground_truth_map = {row['image_id']: name_lookup.get(row['class_name'], "N/A") for _, row in metadata_df.iterrows()}
                    self.status_var.set("Test set labels loaded. Ground truth will be displayed.")
                else:
                    self.status_var.set("Error: Metadata file has no 'dx' or 'class_name' column.")
                    return

            except ImportError:
                messagebox.showerror("Dependency Error", "The 'pandas' library is required for ground truth comparison.\nPlease install it via 'pip install pandas'.")
            except Exception as e:
                self.status_var.set(f"Found metadata, but failed to load: {e}")
        else:
            self.status_var.set("Metadata/labels file not found. Running without ground truth.")

    def load_model_dialog(self):
        model_path = filedialog.askopenfilename(title="Select a Model File", filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*")))
        if not model_path: return
        self.status_var.set(f"Loading model..."); self.root.update_idletasks()
        try:
            self.model, self.model_arch = get_model_architecture(model_path, len(CLASS_NAMES))
            self.model.load_state_dict(torch.load(model_path, map_location=device)); self.model.to(device).eval()
            self.input_size = 299 if 'inception' in self.model_arch.lower() else 224
            self.model_info_label.config(text=f"Architecture: {self.model_arch}\nInput Size: {self.input_size}x{self.input_size}")
            self.select_img_btn.config(state=tk.NORMAL); self.select_folder_btn.config(state=tk.NORMAL)
            self.status_indicator.itemconfig('status', fill=self.SUCCESS_COLOR)
            self.status_var.set("Model loaded. Ready to classify.")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model:\n{e}"); self.status_var.set("Error loading model.")

    def select_image_dialog(self):
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=(('Image files', '*.jpg *.jpeg *.png'), ('All files', '*.*')))
        if image_path: self.process_single_image(image_path)
        
    def handle_drop(self, event):
        if self.model is None: messagebox.showwarning("Warning", "Please load a model first."); return
        filepaths = self.root.tk.splitlist(event.data)
        if os.path.isfile(filepaths[0]): self.process_single_image(filepaths[0])

    def on_tree_double_click(self, event):
        if not self.tree.selection(): return
        item = self.tree.selection()[0]
        filename = self.tree.item(item, 'values')[0]
        if self.batch_folder_path and os.path.exists(os.path.join(self.batch_folder_path, filename)):
            self.process_single_image(os.path.join(self.batch_folder_path, filename))
    
    def process_single_image(self, image_path):
        self.status_var.set(f"Processing {os.path.basename(image_path)}..."); self.root.update_idletasks()
        try:
            pil_image = Image.open(image_path).convert('RGB')
            prediction, all_probs = self.predict(pil_image)
            self.display_image(pil_image)
            self._update_probability_chart(all_probs)
            self.status_var.set(f"Completed: {os.path.basename(image_path)} - Prediction: {prediction.split('(')[0]}")
        except Exception as e:
            messagebox.showerror("Image Processing Error", f"Failed to process image:\n{e}"); self.status_var.set("Error processing image.")

    def _start_batch_process(self):
        folder_path = filedialog.askdirectory(title="Select a Folder of Images")
        if not folder_path: return
        
        self._load_metadata(folder_path)
        
        self.batch_folder_path = folder_path
        self.tree.delete(*self.tree.get_children())
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files: messagebox.showinfo("Info", "No valid image files found."); return
        
        self.progress['maximum'] = len(image_files); self.progress['value'] = 0
        self.select_folder_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_batch_thread, args=(image_files, folder_path), daemon=True).start()
        self._check_batch_queue()

    def _run_batch_thread(self, image_files, folder_path):
        for i, filename in enumerate(image_files):
            try:
                pil_image = Image.open(os.path.join(folder_path, filename)).convert('RGB')
                prediction, probs = self.predict(pil_image)
                self.result_queue.put(('data', (filename, prediction, f"{probs.max():.2%}", i + 1)))
            except Exception as e:
                self.result_queue.put(('error', (filename, str(e), i + 1)))
        self.result_queue.put(('done', len(image_files)))

    def _check_batch_queue(self):
        try:
            while True:
                msg_type, data = self.result_queue.get_nowait()
                if msg_type == 'data':
                    filename, pred, conf, count = data
                    
                    image_id = os.path.splitext(filename)[0]
                    actual_class = self.ground_truth_map.get(image_id, "N/A")
                    
                    is_correct = (pred == actual_class)
                    correct_symbol = '✔️' if is_correct else '❌'
                    tags = ('correct',) if is_correct else ('incorrect',)
                    
                    self.tree.insert('', 'end', values=(filename, pred, actual_class, correct_symbol, conf), tags=tags)
                    self.progress['value'] = count
                    self.status_var.set(f"Processing file {count}/{self.progress['maximum']}...")
                elif msg_type == 'error':
                    filename, err_msg, count = data
                    self.tree.insert('', 'end', values=(filename, f"Error: {err_msg[:50]}...", "N/A", "N/A", "-"))
                    self.progress['value'] = count
                elif msg_type == 'done':
                    total_files = data
                    self.status_var.set(f"Batch processing complete. Processed {total_files} images.")
                    self.select_folder_btn.config(state=tk.NORMAL)
                    self.progress['value'] = 0
                    return
        except queue.Empty:
            self.root.after(100, self._check_batch_queue)

    def predict(self, pil_image):
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        return CLASS_NAMES[probs.argmax().item()], probs.cpu().numpy()

    def _update_probability_chart(self, probabilities):
        self.ax.clear(); self.ax.set_facecolor(self.BG_COLOR)
        y_pos = np.arange(len(CLASS_NAMES))
        colors = [self.DANGER_COLOR if name in MALIGNANT_CLASSES else self.ACCENT_COLOR for name in CLASS_NAMES]
        self.ax.barh(y_pos, probabilities, align='center', color=colors)
        self.ax.set_yticks(y_pos); self.ax.set_yticklabels([name.split('(')[0].strip() for name in CLASS_NAMES], fontsize=9)
        self.ax.invert_yaxis(); self.ax.set_xlabel('Probability', fontsize=9)
        self.ax.tick_params(axis='x', labelsize=8)
        for spine in ['top', 'right']: self.ax.spines[spine].set_visible(False)
        for index, value in enumerate(probabilities): self.ax.text(value + 0.01, index, f'{value:.1%}', va='center', fontsize=8, color=self.TEXT_COLOR)
        self.ax.set_xlim(0, max(1.0, probabilities.max() * 1.15))
        self.fig.tight_layout(pad=1.5); self.chart_canvas.draw()

    def display_image(self, pil_image):
        self.image_canvas.delete("all")
        w, h = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
        if w <= 1 or h <= 1: w, h = 400, 400
        scale = min((w - 20) / pil_image.width, (h - 20) / pil_image.height)
        nw, nh = int(pil_image.width * scale), int(pil_image.height * scale)
        self.tk_img = ImageTk.PhotoImage(pil_image.resize((nw, nh), Image.Resampling.LANCZOS))
        self.image_canvas.create_image(w / 2, h / 2, image=self.tk_img)

    def _export_to_csv(self):
        if not self.tree.get_children(): messagebox.showinfo("Info", "No data to export."); return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not filepath: return
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Prediction', 'Actual_Class', 'Is_Correct', 'Confidence'])
                for row_id in self.tree.get_children(): 
                    writer.writerow(self.tree.item(row_id)['values'])
            self.status_var.set(f"Results exported successfully.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{e}")

    def _sort_treeview_column(self, col, reverse):
        data = [(self.tree.set(child, col), child) for child in self.tree.get_children('')]
        data.sort(reverse=reverse)
        for index, (val, child) in enumerate(data): self.tree.move(child, '', index)
        self.tree.heading(col, command=lambda: self._sort_treeview_column(col, not reverse))
        
    def _center_drop_hint(self):
        if self.image_canvas.find_withtag("drop_hint"):
            self.image_canvas.coords(self.drop_hint_id, self.image_canvas.winfo_width()/2, self.image_canvas.winfo_height()/2)
            
    def _show_about(self):
        messagebox.showinfo("About Dermatology AI Assistant","Version 3.0\n\nThis application uses deep learning to classify skin lesions.\n\nDisclaimer: This tool is for informational purposes and is NOT a substitute for professional medical advice, diagnosis, or treatment.")

    def _on_closing(self):
        self._save_settings()
        self.root.destroy()

    def _save_settings(self):
        settings = { "geometry": self.root.geometry(), "sash_pos": self.paned_window.sashpos(0) }
        try:
            with open(SETTINGS_FILE, 'w') as f: json.dump(settings, f, indent=4)
        except Exception as e: print(f"Warning: Could not save settings: {e}") 

    def _load_settings(self):
        if not os.path.exists(SETTINGS_FILE): self.root.geometry("1200x800"); return
        try:
            with open(SETTINGS_FILE, 'r') as f: settings = json.load(f)
            if geo := settings.get("geometry"): self.root.geometry(geo)
            if sash_pos := settings.get("sash_pos"):
                self.root.update_idletasks() 
                self.paned_window.sashpos(0, sash_pos)
        except Exception as e:
            print(f"Warning: Could not load settings: {e}"); self.root.geometry("1200x800")


if __name__ == '__main__':
    # Add the environment variable fix for the OMP: Error #15 issue on some systems
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    # Use TkinterDnD.Tk() instead of tk.Tk()
    root = TkinterDnD.Tk()
    root.title("Dermatology AI Assistant")
    
    app = SkinLesionClassifierApp(root)
    root.mainloop()
