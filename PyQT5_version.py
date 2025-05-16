# C:\Users\bshan\AppData\Local\Microsoft\WindowsApps\python3.exe C:\Github\gemini.py
import sys
sys.path.append(r"C:\\Github\\Trump_Trade_Agreements_America_Is_Back_Info\\.venv\\Lib\\site-packages")
import os
import time
import base64
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget, QFileDialog,
    QProgressBar, QMessageBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QSplitter, QGroupBox, QStyleFactory
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QRect

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold, Candidate
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    genai = None
    GenerationConfig = None
    HarmCategory = None
    HarmBlockThreshold = None
    Candidate = None
    print("WARNING: google.generativeai library not found. AI features will be disabled.")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    io = None
    print("WARNING: Pillow (PIL) library not found. Real image display from API will be disabled.")

DEFAULT_API_KEY_PLACEHOLDER = "YOUR_GOOGLE_AI_API_KEY_HERE"
IMAGE_GENERATION_MODEL_CANDIDATE = "imagen-3.0-generate-002"

def flash_2_0_generate_base(base_prompt, width, height):
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(QColor("#111111"))
    painter = QPainter(img)
    painter.setPen(QColor(Qt.white))
    painter.setFont(QFont("Arial", 14, QFont.Bold))
    painter.drawText(QRect(0, 0, width, 30), Qt.AlignCenter, "Flash 2.0 Base Layer")
    painter.setFont(QFont("Arial", 10))
    painter.drawText(QRect(0, 30, width, 60), Qt.AlignCenter, f"Placeholders for: {base_prompt[:50]}")
    # Minimal placeholders
    placeholder_positions = [(50, 100, 100, 100), (200, 250, 120, 120)]
    for i, (x, y, w, h) in enumerate(placeholder_positions):
        painter.drawRect(x, y, w, h)
        painter.drawText(QRect(x, y, w, h), Qt.AlignCenter, f"GEN_{i}")
    painter.end()
    pixmap = QPixmap.fromImage(img)
    return pixmap, placeholder_positions

def flash_2_0_fill_placeholder(placeholder_idx, generator_prompt, w, h):
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor("#444444"))
    painter = QPainter(img)
    painter.setPen(QColor(Qt.white))
    painter.setFont(QFont("Arial", 10))
    painter.drawText(QRect(0, 0, w, h), Qt.AlignCenter, f"Filled {placeholder_idx}\n{generator_prompt[:30]}")
    painter.end()
    return QPixmap.fromImage(img)

def flash_2_0_stitch_layers(base_img, placeholders, filler_images):
    if not PIL_AVAILABLE:
        return base_img
    base_bytes = bytearray()
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    base_img.save(buffer, "PNG")
    base_bytes.extend(buffer.data())
    buffer.close()
    base_pil = Image.open(io.BytesIO(bytes(base_bytes)))
    base_pil = base_pil.convert("RGBA")
    for idx, (pos) in enumerate(placeholders):
        x, y, w, h = pos
        if idx < len(filler_images):
            filler = filler_images[idx]
            filler_bytes = bytearray()
            buf_f = QBuffer()
            buf_f.open(QIODevice.WriteOnly)
            filler.save(buf_f, "PNG")
            filler_bytes.extend(buf_f.data())
            buf_f.close()
            filler_pil = Image.open(io.BytesIO(bytes(filler_bytes))).convert("RGBA").resize((w, h), Image.LANCZOS)
            base_pil.alpha_composite(filler_pil, (x, y))
    final_bytes = io.BytesIO()
    base_pil.save(final_bytes, format="PNG")
    final_qimg = QImage.fromData(final_bytes.getvalue(), "PNG")
    return QPixmap.fromImage(final_qimg)

class ImageGenerationWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, api_key, user_prompt, generation_params, attempt_real_image=True):
        super().__init__()
        self.api_key = api_key
        self.user_prompt = user_prompt
        self.generation_params = generation_params
        self.attempt_real_image = attempt_real_image

    def run(self):
        if not GOOGLE_GENAI_AVAILABLE:
            self.progress.emit("Google GenAI library not available.")
            self._simulate_image("AI Library Missing")
            return
        if not self.api_key or self.api_key == DEFAULT_API_KEY_PLACEHOLDER:
            self.progress.emit("API Key not configured or is placeholder (worker check).")
            self.finished.emit("Error: API Key missing or invalid.")
            return
        try:
            model_name = self.generation_params.get("model_name", IMAGE_GENERATION_MODEL_CANDIDATE)
            self.progress.emit(f"Selected model: {model_name}")
            safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            ]
            is_image_gen_model = "imagen" in model_name or "image" in model_name
            if self.attempt_real_image and is_image_gen_model and PIL_AVAILABLE:
                self.progress.emit(f"Attempting REAL image generation with {model_name}")
                cfg_dict = {
                    "temperature": self.generation_params.get("temperature", 0.9),
                    "top_p": self.generation_params.get("top_p", 1.0),
                    "top_k": self.generation_params.get("top_k", None),
                    "candidate_count": 1,
                }
                if cfg_dict["top_k"] is None:
                    del cfg_dict["top_k"]
                final_conf = GenerationConfig(**{k: v for k, v in cfg_dict.items() if v is not None})
                image_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=final_conf,
                    safety_settings=safety_settings
                )
                try:
                    self.progress.emit("Calling generate_content...")
                    response = image_model.generate_content(self.user_prompt)
                    self.progress.emit("Call done.")
                except Exception as gen_e:
                    self.progress.emit(f"Error: {gen_e}")
                    self._simulate_image("API Error")
                    return
                img_bytes = None
                if not response.candidates:
                    self.progress.emit("No candidates.")
                    self._simulate_image("No candidates")
                    return
                for candidate_idx, candidate in enumerate(response.candidates):
                    if candidate.content and candidate.content.parts:
                        for part_idx, part in enumerate(candidate.content.parts):
                            mt = getattr(part, 'mime_type', '')
                            if mt.startswith("image/"):
                                if hasattr(part, 'inline_data') and part.inline_data and getattr(part.inline_data, 'data', None):
                                    img_bytes = part.inline_data.data
                                    break
                        if img_bytes:
                            break
                if img_bytes:
                    try:
                        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                        data = pil_image.tobytes("raw", "RGBA")
                        q_image = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
                        if q_image.isNull():
                            self._simulate_image("Null QImage")
                            return
                        pm = QPixmap.fromImage(q_image)
                        if pm.isNull():
                            self._simulate_image("Null QPixmap")
                            return
                        # Fusion approach with placeholders
                        base_pixmap, placeholders = flash_2_0_generate_base(self.user_prompt, pm.width(), pm.height())
                        filled_images = []
                        for idx, coords in enumerate(placeholders):
                            fill_pm = flash_2_0_fill_placeholder(idx, self.user_prompt, coords[2], coords[3])
                            filled_images.append(fill_pm)
                        stitched = flash_2_0_stitch_layers(pm, placeholders, filled_images)
                        self.finished.emit(stitched)
                        return
                    except Exception as img_e:
                        self.progress.emit(f"Error processing: {img_e}")
                        self._simulate_image("Image Processing Error")
                        return
                else:
                    self._simulate_image("No image data.")
                    return
            elif not PIL_AVAILABLE and self.attempt_real_image and is_image_gen_model:
                self.progress.emit("Pillow missing.")
                self._simulate_image("Pillow Missing")
                return
            else:
                cfg_dict = {
                    "temperature": self.generation_params.get("temperature", 0.9),
                    "top_p": self.generation_params.get("top_p", 1.0),
                    "top_k": self.generation_params.get("top_k", None),
                    "max_output_tokens": self.generation_params.get("max_output_tokens", 2048),
                }
                if cfg_dict["top_k"] is None:
                    del cfg_dict["top_k"]
                text_conf = GenerationConfig(**{k: v for k, v in cfg_dict.items() if v is not None})
                prompt_text = (
                    f"Generate a scene description for '{self.user_prompt}'."
                )
                if is_image_gen_model:
                    model_name = "gemini-1.5-flash-latest"
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=text_conf,
                    safety_settings=safety_settings
                )
                resp = model.generate_content(prompt_text)
                desc = ""
                if hasattr(resp, 'text') and resp.text:
                    desc = resp.text
                elif hasattr(resp, 'candidates') and resp.candidates:
                    for c in resp.candidates:
                        if c.content and c.content.parts:
                            for part in c.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    desc = part.text
                                    break
                            if desc:
                                break
                if not desc:
                    self._simulate_image("No text content")
                    return
                # Flash 2.0 approach in simulation
                base_pm, placeholders = flash_2_0_generate_base(self.user_prompt, self.generation_params.get("target_image_width", 512), self.generation_params.get("target_image_height", 512))
                filled_images = []
                for idx, coords in enumerate(placeholders):
                    fill_pm = flash_2_0_fill_placeholder(idx, desc, coords[2], coords[3])
                    filled_images.append(fill_pm)
                stitched = flash_2_0_stitch_layers(base_pm, placeholders, filled_images)
                self.finished.emit(stitched)
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self._simulate_image("Critical Error")

    def _simulate_image(self, text="Simulated", desc=None):
        w = self.generation_params.get("target_image_width", 512)
        h = self.generation_params.get("target_image_height", 512)
        img = QImage(w, h, QImage.Format_RGB32)
        img.fill(QColor("#2E2E2E"))
        p = QPainter(img)
        p.setPen(QColor(Qt.white))
        p.setFont(QFont("Arial", 14, QFont.Bold))
        p.drawText(QRect(10, 10, w-20, 60), Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, text)
        p.setPen(QColor("#AAAAAA"))
        p.setFont(QFont("Arial", 9, QFont.StyleItalic))
        pr = f"User Idea: {self.user_prompt[:150]}"
        p.drawText(QRect(15, 75, w-30, 40), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, pr)
        off = 125
        if desc:
            p.setPen(QColor("#DDDDDD"))
            p.setFont(QFont("Arial", 8))
            p.drawText(QRect(15, off, w-30, h-off-10), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, f"Desc:\n{desc}")
        p.end()
        pm = QPixmap.fromImage(img)
        self.finished.emit(pm)

from PyQt5.QtCore import QIODevice, QBuffer

class SmartImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Instructional Image Generation Tool (Imagen/Gemini)")
        self.setGeometry(50, 50, 950, 750)
        self.settings = QSettings("MyCompany", "SmartImageAppImagenGemini")
        self.api_key = ""
        self.genai_configured_successfully = False
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self.user_request_tab = QWidget()
        self.system_settings_tab = QWidget()
        self.developer_pipeline_tab = QWidget()
        self.tabs.addTab(self.user_request_tab, "🎨 Image Generation")
        self.tabs.addTab(self.system_settings_tab, "⚙️ System Settings")
        self.tabs.addTab(self.developer_pipeline_tab, "🛠️ Developer & Pipeline")
        self._init_user_request_ui()
        self._init_system_settings_ui()
        self._init_developer_pipeline_ui()
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Application Ready. Configure API Key in System Settings.")
        self.current_pixmap = None
        self.worker = None
        self.load_settings()
        self.check_libraries_status()

    def check_libraries_status(self):
        if not GOOGLE_GENAI_AVAILABLE:
            QMessageBox.warning(self, "Library Missing", "The 'google-generativeai' library is missing.")
        if not PIL_AVAILABLE:
            QMessageBox.warning(self, "Library Missing", "The 'Pillow' library is missing.")

    def _init_user_request_ui(self):
        layout = QVBoxLayout(self.user_request_tab)
        main_splitter = QSplitter(Qt.Horizontal)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        self.prompt_label = QLabel("<b>Enter Your Image Idea/Prompt:</b>")
        controls_layout.addWidget(self.prompt_label)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("e.g., 'A futuristic cityscape at dusk...'")
        self.prompt_input.setFixedHeight(120)
        controls_layout.addWidget(self.prompt_input)
        params_group = QGroupBox("Generation Parameters")
        params_form_layout = QFormLayout()
        self.img_width_input = QSpinBox()
        self.img_width_input.setRange(256, 4096); self.img_width_input.setValue(1024); self.img_width_input.setSuffix(" px")
        params_form_layout.addRow("Sim. Image Width (display):", self.img_width_input)
        self.img_height_input = QSpinBox()
        self.img_height_input.setRange(256, 4096); self.img_height_input.setValue(1024); self.img_height_input.setSuffix(" px")
        params_form_layout.addRow("Sim. Image Height (display):", self.img_height_input)
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 2.0); self.temperature_input.setValue(0.9); self.temperature_input.setSingleStep(0.05)
        params_form_layout.addRow("Temperature (0.0-2.0):", self.temperature_input)
        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setRange(50, 8192); self.max_tokens_input.setValue(2048)
        params_form_layout.addRow("Max Tokens (Text Fallback):", self.max_tokens_input)
        self.style_input = QLineEdit()
        self.style_input.setPlaceholderText("e.g., photorealistic, watercolor, cinematic")
        params_form_layout.addRow("Desired Style (appended):", self.style_input)
        params_group.setLayout(params_form_layout)
        controls_layout.addWidget(params_group)
        self.generate_button = QPushButton("✨ Generate Image")
        self.generate_button.setFixedHeight(40)
        self.generate_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 16px; border-radius: 5px; } QPushButton:hover { background-color: #45a049; }")
        self.generate_button.clicked.connect(self.start_image_generation)
        controls_layout.addWidget(self.generate_button)
        self.save_image_button = QPushButton("💾 Save Image")
        self.save_image_button.setEnabled(False)
        self.save_image_button.clicked.connect(self.save_current_image)
        controls_layout.addWidget(self.save_image_button)
        controls_layout.addStretch()
        main_splitter.addWidget(controls_widget)
        display_log_widget = QWidget()
        display_log_layout = QVBoxLayout(display_log_widget)
        self.image_display_label = QLabel("Your generated image will appear here.")
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setMinimumSize(400, 300)
        self.image_display_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0; border-radius: 5px;")
        display_log_layout.addWidget(self.image_display_label, 3)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        display_log_layout.addWidget(self.progress_bar)
        log_group = QGroupBox("Generation Log (User)")
        log_layout_inner = QVBoxLayout()
        self.generation_log = QTextEdit()
        self.generation_log.setReadOnly(True)
        self.generation_log.setFixedHeight(150)
        self.generation_log.setPlaceholderText("Follow the image generation process here...")
        log_layout_inner.addWidget(self.generation_log)
        log_group.setLayout(log_layout_inner)
        display_log_layout.addWidget(log_group, 1)
        main_splitter.addWidget(display_log_widget)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)
        layout.addWidget(main_splitter)

    def _init_system_settings_ui(self):
        layout = QVBoxLayout(self.system_settings_tab)
        form_layout = QFormLayout()
        self.api_key_label = QLabel("Google AI API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow(self.api_key_label, self.api_key_input)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText(f"e.g., {IMAGE_GENERATION_MODEL_CANDIDATE}")
        form_layout.addRow("AI Model Name:", self.model_name_input)
        layout.addLayout(form_layout)
        self.save_settings_button = QPushButton("Save Settings & Apply API Key")
        self.save_settings_button.clicked.connect(self.save_and_apply_settings)
        layout.addWidget(self.save_settings_button)
        info = QLabel("<small><i>Configure your API key and model. Pillow required for real image generation.</i></small>")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()

    def _init_developer_pipeline_ui(self):
        layout = QVBoxLayout(self.developer_pipeline_tab)
        info_label = QLabel(
            "<h3>Developer & Pipeline Information</h3>"
            "<b>Fusion with Flash 2.0:</b> First generate a base layer with placeholders. Then fill placeholders individually and stitch."
        )
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText)
        layout.addWidget(info_label)
        dev_log_group = QGroupBox("Detailed Developer Log")
        dev_log_layout_inner = QVBoxLayout()
        self.dev_log_display = QTextEdit()
        self.dev_log_display.setReadOnly(True)
        self.dev_log_display.setPlaceholderText("Developer-level logs...")
        self.dev_log_display.setFont(QFont("Consolas", 8))
        dev_log_layout_inner.addWidget(self.dev_log_display)
        dev_log_group.setLayout(dev_log_layout_inner)
        layout.addWidget(dev_log_group, 1)

    def load_settings(self):
        loaded_api_key = self.settings.value("api_key", DEFAULT_API_KEY_PLACEHOLDER)
        self.api_key = loaded_api_key
        if loaded_api_key == DEFAULT_API_KEY_PLACEHOLDER:
            self.api_key_input.setText("")
            self.api_key_input.setPlaceholderText("Enter API Key")
        else:
            self.api_key_input.setText(loaded_api_key)
        default_model_name = IMAGE_GENERATION_MODEL_CANDIDATE if GOOGLE_GENAI_AVAILABLE else 'N/A'
        self.model_name_input.setText(self.settings.value("ai_model_name", default_model_name))
        self.img_width_input.setValue(int(self.settings.value("user_img_width", 1024)))
        self.img_height_input.setValue(int(self.settings.value("user_img_height", 1024)))
        self.temperature_input.setValue(float(self.settings.value("user_temperature", 0.9)))
        self.max_tokens_input.setValue(int(self.settings.value("user_max_tokens", 2048)))
        self.prompt_input.setText(self.settings.value("user_last_prompt", ""))
        self.style_input.setText(self.settings.value("user_last_style", ""))
        msg = "Settings loaded."
        if self.api_key and self.api_key != DEFAULT_API_KEY_PLACEHOLDER:
            msg += " Attempting AI config..."
            self.status_bar.showMessage(msg)
            if not self.configure_genai_globally(self.api_key):
                self.status_bar.showMessage("Failed to configure AI. Check settings.", 10000)
        else:
            msg += " API Key not set."
            self.status_bar.showMessage(msg, 10000)
        self.update_dev_log("App loaded. " + msg)

    def configure_genai_globally(self, key_to_try=None):
        self.genai_configured_successfully = False
        if not GOOGLE_GENAI_AVAILABLE:
            m = "Google GenAI not found."
            self.update_dev_log(m)
            self.status_bar.showMessage(m, 10000)
            return False
        if key_to_try is None:
            key_to_try = self.api_key_input.text().strip()
            if not key_to_try or key_to_try == DEFAULT_API_KEY_PLACEHOLDER:
                if self.api_key and self.api_key != DEFAULT_API_KEY_PLACEHOLDER:
                    key_to_try = self.api_key
                else:
                    key_to_try = None
        if key_to_try and key_to_try != DEFAULT_API_KEY_PLACEHOLDER:
            try:
                self.update_dev_log(f"Configuring genai with key: {'*'*(len(key_to_try)-4)+key_to_try[-4:]}")
                genai.configure(api_key=key_to_try)
                test_model_name = self.model_name_input.text().strip()
                if not test_model_name:
                    test_model_name = IMAGE_GENERATION_MODEL_CANDIDATE
                self.update_dev_log(f"Testing access to model: {test_model_name}")
                genai.get_model(test_model_name)
                m = f"GenAI configured. Model '{test_model_name}' accessible."
                self.update_dev_log(m)
                self.status_bar.showMessage(m, 7000)
                self.api_key = key_to_try
                self.genai_configured_successfully = True
                return True
            except Exception as e:
                er = f"Error: {e}"
                QMessageBox.warning(self, "API Key/Model Error", er)
                self.status_bar.showMessage("GenAI Config Error", 10000)
                self.update_dev_log(er)
                return False
        else:
            m = "API key is placeholder."
            self.update_dev_log(m)
            self.status_bar.showMessage(m, 7000)
            return False

    def save_and_apply_settings(self):
        new_api_key = self.api_key_input.text().strip()
        current_model = self.model_name_input.text().strip()
        if not new_api_key:
            if self.api_key and self.api_key != DEFAULT_API_KEY_PLACEHOLDER:
                new_api_key = self.api_key
            else:
                new_api_key = DEFAULT_API_KEY_PLACEHOLDER
        self.settings.setValue("api_key", new_api_key)
        self.settings.setValue("ai_model_name", current_model)
        self.settings.setValue("user_img_width", self.img_width_input.value())
        self.settings.setValue("user_img_height", self.img_height_input.value())
        self.settings.setValue("user_temperature", self.temperature_input.value())
        self.settings.setValue("user_max_tokens", self.max_tokens_input.value())
        self.update_dev_log("Settings saved.")
        c = self.configure_genai_globally(new_api_key)
        if c:
            QMessageBox.information(self, "Settings Applied", "API Key applied.")
            self.status_bar.showMessage("Settings saved, AI configured.", 5000)
        else:
            self.status_bar.showMessage("Settings saved, but AI config failed.", 7000)

    def update_generation_log(self, message):
        self.generation_log.append(message)
        QApplication.processEvents()

    def update_dev_log(self, message):
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.dev_log_display.append(f"[{t}] {message}")
        QApplication.processEvents()

    def start_image_generation(self):
        prompt_base = self.prompt_input.toPlainText().strip()
        style_suffix = self.style_input.text().strip()
        if not prompt_base:
            QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
            return
        final_prompt = prompt_base
        if style_suffix:
            final_prompt = f"{prompt_base}, style: {style_suffix}"
        self.update_generation_log(f"User Prompt: {final_prompt[:150]}")
        if not GOOGLE_GENAI_AVAILABLE:
            self.update_generation_log("Google AI library missing. Simulating.")
            self._handle_simulation_fallback("AI Lib Missing")
            return
        if not self.genai_configured_successfully:
            QMessageBox.warning(self, "Config Error", "AI not configured. Check System Settings.")
            self.tabs.setCurrentWidget(self.system_settings_tab)
            return
        current_model_name = self.model_name_input.text().strip()
        if not current_model_name:
            QMessageBox.warning(self, "Config Error", "Model name not set.")
            self.tabs.setCurrentWidget(self.system_settings_tab)
            return
        is_image = ("imagen" in current_model_name or "image" in current_model_name)
        if is_image and not PIL_AVAILABLE:
            QMessageBox.warning(self, "Library Missing", "Pillow is missing. Will fallback.")
        self.generate_button.setEnabled(False)
        self.save_image_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.image_display_label.setText("AI is thinking...")
        self.image_display_label.setPixmap(QPixmap())
        self.update_generation_log(f"Starting with model: {current_model_name}")
        params = {
            "model_name": current_model_name,
            "temperature": self.temperature_input.value(),
            "max_output_tokens": self.max_tokens_input.value(),
            "top_p": self.temperature_input.value() + 0.1 if self.temperature_input.value() < 0.9 else 1.0,
            "top_k": 40 if self.temperature_input.value() > 0.5 else None,
            "target_image_width": self.img_width_input.value(),
            "target_image_height": self.img_height_input.value(),
        }
        self.update_dev_log(f"Params: {params}")
        self.worker = ImageGenerationWorker(self.api_key, final_prompt, params, attempt_real_image=True)
        self.worker.progress.connect(lambda m: self.update_dev_log(f"[Worker] {m}"))
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.start()

    def _handle_simulation_fallback(self, reason="Fallback"):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.generate_button.setEnabled(True)
        self.image_display_label.setText(f"{reason} placeholder.")
        dummy = QImage(self.img_width_input.value(), self.img_height_input.value(), QImage.Format_RGB32)
        dummy.fill(Qt.darkGray)
        paint = QPainter(dummy)
        paint.setPen(QColor(Qt.white))
        paint.setFont(QFont("Arial", 12, QFont.Bold))
        paint.drawText(QRect(0, 0, dummy.width(), dummy.height()//2), Qt.AlignCenter, reason)
        paint.setFont(QFont("Arial", 8))
        txt = self.prompt_input.toPlainText()
        paint.drawText(QRect(0, dummy.height()//2, dummy.width(), dummy.height()//2), Qt.AlignCenter, f"Prompt: {txt[:100]}")
        paint.end()
        self.current_pixmap = QPixmap.fromImage(dummy)
        self._display_scaled_pixmap()
        self.save_image_button.setEnabled(True)
        self.update_generation_log(f"{reason} used for {txt[:50]}")

    def on_generation_finished(self, result):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.generate_button.setEnabled(True)
        if isinstance(result, QPixmap):
            self.current_pixmap = result
            if self.current_pixmap.isNull():
                self.update_generation_log("Got null QPixmap.")
                self._handle_simulation_fallback("Null QPixmap")
            else:
                self._display_scaled_pixmap()
                self.save_image_button.setEnabled(True)
                self.update_generation_log("Success!")
                self.status_bar.showMessage("Image ready!", 5000)
        elif isinstance(result, str) and result.startswith("Error:"):
            e = result
            self.image_display_label.setText(f"Error.\n{e.split(':',1)[1].strip()}")
            self.update_generation_log(e)
            self.current_pixmap = None
            c = e.split(':',1)[1][:100].strip() if ':' in e else e[:100]
            self._handle_simulation_fallback(f"Error: {c}")
        else:
            self.image_display_label.setText("Unexpected result.")
            self.update_generation_log(f"Unexpected result {type(result)}")
            self.current_pixmap = None
            self._handle_simulation_fallback("Unexpected")

    def _display_scaled_pixmap(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            w = self.image_display_label.width()
            h = self.image_display_label.height()
            if w <= 10 or h <= 10:
                self.image_display_label.setPixmap(self.current_pixmap)
                return
            scaled = self.current_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display_label.setPixmap(scaled)
        else:
            if not self.generate_button.isEnabled():
                self.image_display_label.setText("AI is thinking...")
            else:
                self.image_display_label.setText("No image.")

    def save_current_image(self):
        if not self.current_pixmap or self.current_pixmap.isNull():
            QMessageBox.warning(self, "Save Error", "No valid image to save.")
            return
        prompt_text = self.prompt_input.toPlainText()[:30].strip()
        sanitized = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt_text)
        default_filename = f"gemini_art_{sanitized}.png" if sanitized else "gemini_generated_image.png"
        save_dir = os.path.join(os.path.expanduser("~"), "Pictures", "GeminiSmartAppImages")
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except OSError:
                save_dir = os.path.expanduser("~")
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.join(save_dir, default_filename), "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)")
        if filePath:
            if self.current_pixmap.save(filePath):
                self.status_bar.showMessage(f"Saved to {filePath}", 5000)
                self.update_generation_log(f"Saved: {filePath}")
            else:
                QMessageBox.critical(self, "Save Error", f"Could not save to {filePath}.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'image_display_label') and self.image_display_label.isVisible():
            if self.current_pixmap and not self.current_pixmap.isNull():
                self._display_scaled_pixmap()

    def closeEvent(self, event):
        self.update_dev_log("Closing. Saving settings.")
        self.settings.setValue("user_last_prompt", self.prompt_input.toPlainText())
        self.settings.setValue("user_last_style", self.style_input.text())
        self.settings.sync()
        if self.worker and self.worker.isRunning():
            self.update_dev_log("Stopping worker...")
            self.worker.quit()
            if not self.worker.wait(1500):
                self.update_dev_log("Forcing terminate.")
                self.worker.terminate()
                self.worker.wait()
            else:
                self.update_dev_log("Worker stopped.")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if "Fusion" in QApplication.style().objectName() or "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    main_window = SmartImageApp()
    main_window.show()
    sys.exit(app.exec_())
