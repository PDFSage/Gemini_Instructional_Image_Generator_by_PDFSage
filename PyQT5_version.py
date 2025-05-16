
# C:\Users\bshan\AppData\Local\Microsoft\WindowsApps\python3.exe C:\Github\gemini.py
import sys
import os
import time
import base64
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget, QFileDialog,
    QProgressBar, QMessageBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QSplitter, QGroupBox, QStyleFactory  # <--- Add QStyleFactory here
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
    Candidate = None # Add Candidate here for type hinting if needed elsewhere
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
IMAGE_GENERATION_MODEL_CANDIDATE = "imagen-3.0-generate-002" # Example, ensure this model is available via Gemini API

class ImageGenerationWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, api_key, user_prompt, generation_params, attempt_real_image=True):
        super().__init__()
        self.api_key = api_key # Retained for context, but global genai.configure is primary
        self.user_prompt = user_prompt
        self.generation_params = generation_params
        self.attempt_real_image = attempt_real_image

    def run(self):
        if not GOOGLE_GENAI_AVAILABLE:
            self.progress.emit("Google GenAI library not available.")
            self._simulate_image("AI Library Missing (Offline Simulation)")
            return

        # API key check is done by configure_genai_globally before worker starts in practice.
        # This is a redundant check, but harmless.
        if not self.api_key or self.api_key == DEFAULT_API_KEY_PLACEHOLDER:
            self.progress.emit("API Key not configured or is placeholder (worker check).")
            # The main app flow should prevent this if genai_configured_successfully is false
            self.finished.emit("Error: API Key missing or invalid. Please set it in System Settings.")
            return

        try:
            model_name = self.generation_params.get("model_name", IMAGE_GENERATION_MODEL_CANDIDATE)
            self.progress.emit(f"Selected model: {model_name}")

            # Common safety settings
            safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            ]

            is_image_gen_model = "imagen" in model_name or "image" in model_name # Adjusted heuristic

            if self.attempt_real_image and is_image_gen_model and PIL_AVAILABLE:
                self.progress.emit(f"Attempting REAL image generation with {model_name} for: '{self.user_prompt[:100]}...'")

                # Configuration for image generation model
                image_generation_config_dict = {
                    "temperature": self.generation_params.get("temperature", 0.9),
                    "top_p": self.generation_params.get("top_p", 1.0),
                    "top_k": self.generation_params.get("top_k", None),
                    "candidate_count": 1, # Usually want one image
                    # "number_of_images_to_generate": 1 # Some models might use this, check docs for specific model
                }
                if image_generation_config_dict["top_k"] is None:
                    del image_generation_config_dict["top_k"]

                final_image_gen_config = GenerationConfig(
                    **{k: v for k, v in image_generation_config_dict.items() if v is not None}
                )
                self.progress.emit(f"Image Gen Config: {final_image_gen_config}")

                image_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=final_image_gen_config,
                    safety_settings=safety_settings
                )
                self.progress.emit(f"GenerativeModel instance created for '{model_name}'.")

                try:
                    self.progress.emit("Calling generate_content...")
                    # Ensure prompt is just the string, or structure as per model docs if needed
                    response = image_model.generate_content(self.user_prompt)
                    self.progress.emit("generate_content call completed.")
                except Exception as gen_e:
                    self.progress.emit(f"Error during generate_content call: {type(gen_e).__name__}: {gen_e}")
                    self._simulate_image(f"API Call Error: {type(gen_e).__name__}")
                    return

                img_bytes = None
                if not response.candidates:
                    self.progress.emit("API response received, but it contains no candidates.")
                    error_details = "No candidates in response."
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        error_details += f" Prompt Feedback: {response.prompt_feedback}"
                    self._simulate_image(f"No Image: {error_details}")
                    return

                for candidate_idx, candidate in enumerate(response.candidates):
                    self.progress.emit(f"Processing candidate {candidate_idx}. Finish Reason: {candidate.finish_reason}")

                    # Check for non-ideal finish reasons
                    if candidate.finish_reason not in [Candidate.FinishReason.STOP, Candidate.FinishReason.FINISH_REASON_UNSPECIFIED, None, 0]:
                        self.progress.emit(f"Warning: Candidate {candidate_idx} finish reason is '{candidate.finish_reason}'. Image may be affected.")
                    
                    if hasattr(candidate, 'safety_ratings'):
                        for rating_idx, rating in enumerate(candidate.safety_ratings):
                            self.progress.emit(f"  Candidate {candidate_idx} Safety Rating {rating_idx} - Category: {rating.category}, Probability: {rating.probability}, Blocked: {getattr(rating, 'blocked', 'N/A')}")
                            if getattr(rating, 'blocked', False):
                                self.progress.emit(f"    SAFETY BLOCK: Candidate {candidate_idx} content blocked by category {rating.category}")


                    if candidate.content and candidate.content.parts:
                        for part_idx, part in enumerate(candidate.content.parts):
                            part_mime_type = getattr(part, 'mime_type', 'N/A')
                            self.progress.emit(f"  Candidate {candidate_idx}, Part {part_idx}, MIME Type: {part_mime_type}")
                            if part_mime_type and part_mime_type.startswith("image/"):
                                if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data') and part.inline_data.data:
                                    img_bytes = part.inline_data.data
                                    self.progress.emit(f"    Image data found in candidate {candidate_idx}, part {part_idx}.")
                                    break 
                                else:
                                    self.progress.emit(f"    Part has image MIME type but no valid inline_data.data: {part_mime_type}")
                        if img_bytes:
                            break 
                
                if img_bytes:
                    try:
                        pil_image = Image.open(io.BytesIO(img_bytes))
                        data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
                        q_image = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
                        if q_image.isNull():
                            self.progress.emit("QImage conversion resulted in a null image.")
                            self._simulate_image("Error: Image Conversion Failed (QImage null)")
                            return
                        final_pixmap = QPixmap.fromImage(q_image)
                        if final_pixmap.isNull():
                            self.progress.emit("QPixmap conversion resulted in a null pixmap.")
                            self._simulate_image("Error: Image Conversion Failed (QPixmap null)")
                            return
                        self.finished.emit(final_pixmap)
                        self.progress.emit("Real image successfully processed and emitted.")
                        return
                    except Exception as img_e:
                        self.progress.emit(f"Error processing image data: {type(img_e).__name__}: {img_e}")
                        self._simulate_image(f"Image Data Processing Error: {img_e}")
                        return
                else:
                    self.progress.emit("API response processed, but no usable image data extracted from parts.")
                    error_details = "No image data in parts."
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback :
                         error_details += f" Prompt Feedback: {response.prompt_feedback}"
                         for rating in response.prompt_feedback.safety_ratings: # Overall prompt feedback
                            if rating.blocked:
                                error_details += f" Blocked due to {rating.category} at prompt level."
                    self._simulate_image(f"No Image in API Response. {error_details}")
                    return

            elif not PIL_AVAILABLE and self.attempt_real_image and is_image_gen_model:
                self.progress.emit("Pillow library not available. Cannot process real image from API.")
                self._simulate_image("Pillow Missing - Real Image Disabled")
                return
            else: # Fallback to text model for description, or if not an image model
                self.progress.emit(f"Model '{model_name}' not identified as image gen model, or PIL missing. Simulating with text enhancement.")
                
                # Config for text model
                text_model_config_params_dict = {
                    "temperature": self.generation_params.get("temperature", 0.9),
                    "top_p": self.generation_params.get("top_p", 1.0),
                    "top_k": self.generation_params.get("top_k", None),
                    "max_output_tokens": self.generation_params.get("max_output_tokens", 2048), # Text models need this
                }
                if text_model_config_params_dict["top_k"] is None:
                    del text_model_config_params_dict["top_k"]
                
                text_model_generation_config = GenerationConfig(
                    **{k: v for k, v in text_model_config_params_dict.items() if v is not None}
                )
                self.progress.emit(f"Text Model Gen Config: {text_model_generation_config}")

                text_prompt_for_llm = (
                    f"You are an AI assistant that generates detailed, vivid, and creative "
                    f"scene descriptions for an AI image generator. Based on the user's idea: "
                    f"'{self.user_prompt}', create a rich description. Focus on visual elements, "
                    f"atmosphere, style, composition, lighting, and any specific objects or characters mentioned."
                )
                # Use a known text model if current model_name is for images but we are in fallback
                text_model_name_to_use = model_name
                if is_image_gen_model: # If we fell back from an image model due to PIL etc.
                    text_model_name_to_use = "gemini-1.5-flash-latest" # or another suitable text model
                    self.progress.emit(f"Switched to text model '{text_model_name_to_use}' for description generation.")


                model = genai.GenerativeModel(
                    model_name=text_model_name_to_use,
                    generation_config=text_model_generation_config,
                    safety_settings=safety_settings
                )
                text_response = model.generate_content(text_prompt_for_llm) # generation_config is on model
                
                detailed_description = ""
                if hasattr(text_response, 'text') and text_response.text:
                    detailed_description = text_response.text
                elif hasattr(text_response, 'candidates') and text_response.candidates:
                    for candidate in text_response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    detailed_description = part.text
                                    break
                            if detailed_description:
                                break
                
                if not detailed_description:
                    error_info_str = "No content generated by text model."
                    if hasattr(text_response, 'prompt_feedback') and text_response.prompt_feedback:
                        error_info_str += f" Prompt Feedback: {text_response.prompt_feedback}"
                    self._simulate_image(f"Error: AI (text) model returned no content. {error_info_str}")
                    return
                self._simulate_image("Simulated Image from AI Text", detailed_description)

        except Exception as e:
            self.progress.emit(f"Unhandled error during AI processing: {type(e).__name__}: {e}")
            import traceback
            self.progress.emit(f"Traceback: {traceback.format_exc()}")
            self.finished.emit(f"Error: {type(e).__name__}: {e}")
            self._simulate_image(f"Critical Error During API Call: {type(e).__name__}")

    def _simulate_image(self, title_text="Simulated Image", detailed_description=None):
        img_width = self.generation_params.get("target_image_width", 512)
        img_height = self.generation_params.get("target_image_height", 512)
        image = QImage(img_width, img_height, QImage.Format_RGB32)
        image.fill(QColor("#2E2E2E")) # Dark gray background
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Title
        painter.setPen(QColor(Qt.white))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        title_rect = QRect(10, 10, img_width - 20, 60)
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, title_text)
        
        # User Prompt
        painter.setPen(QColor("#AAAAAA")) # Lighter gray for subtitles
        painter.setFont(QFont("Arial", 9, QFont.StyleItalic))
        prompt_display_text = f"User Idea: {self.user_prompt[:150]}" + ("..." if len(self.user_prompt) > 150 else "")
        prompt_rect = QRect(15, 75, img_width - 30, 40) # Adjusted y-offset and height
        painter.drawText(prompt_rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, prompt_display_text)
        
        current_y_offset = 125 # Start Y for description

        # Detailed Description (if available)
        if detailed_description:
            painter.setPen(QColor("#DDDDDD")) # Light color for description text
            painter.setFont(QFont("Arial", 8))
            desc_rect = QRect(15, current_y_offset, img_width - 30, img_height - current_y_offset - 10) # Max height for desc
            painter.drawText(desc_rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, f"Enhanced Description:\n{detailed_description}")
        
        painter.end()
        final_pixmap = QPixmap.fromImage(image)
        self.finished.emit(final_pixmap)

# The rest of the SmartImageApp class and other UI components would remain the same.
# Ensure that `Candidate` is imported from `google.generativeai.types` at the top.
# (This was added in the try-except block for imports)

# SmartImageApp class definition remains here... (it's long, so not repeated)
# if __name__ == '__main__': ... block also remains

class SmartImageApp(QMainWindow): # Keep existing SmartImageApp code
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Instructional Image Generation Tool (Imagen/Gemini)")
        self.setGeometry(50, 50, 950, 750)
        self.settings = QSettings("MyCompany", "SmartImageAppImagenGemini") # Updated app name in settings
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
        self._init_developer_pipeline_ui() # Update info here if needed
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Application Ready. Configure API Key in System Settings.")
        self.current_pixmap = None
        self.worker = None
        self.load_settings()
        self.check_libraries_status()


    def check_libraries_status(self):
        if not GOOGLE_GENAI_AVAILABLE:
            QMessageBox.warning(self, "Library Missing", "The 'google-generativeai' library is not found. AI features will be disabled. Please install it: pip install google-generativeai")
        if not PIL_AVAILABLE:
            QMessageBox.warning(self, "Library Missing", "The 'Pillow' (PIL) library is not found. Displaying real images from the API will not work. Please install it: pip install Pillow")

    def _init_user_request_ui(self):
        layout = QVBoxLayout(self.user_request_tab)
        main_splitter = QSplitter(Qt.Horizontal)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        self.prompt_label = QLabel("<b>Enter Your Image Idea/Prompt:</b>")
        controls_layout.addWidget(self.prompt_label)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("e.g., 'A futuristic cityscape at dusk...' or 'A cute cat astronaut exploring Mars.'")
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
        params_form_layout.addRow("Temperature (0.0-2.0):", self.temperature_input) # Label clarity
        self.max_tokens_input = QSpinBox() # This is for text models in fallback
        self.max_tokens_input.setRange(50, 8192); self.max_tokens_input.setValue(2048)
        params_form_layout.addRow("Max Tokens (Text Fallback):", self.max_tokens_input) # Label clarity
        self.style_input = QLineEdit()
        self.style_input.setPlaceholderText("e.g., photorealistic, watercolor, pixel art, cinematic (appended to prompt if filled)")
        params_form_layout.addRow("Desired Style (appended):", self.style_input) # Label clarity
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
        display_log_layout.addWidget(self.image_display_label, 3) # Weight 3 for image
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        display_log_layout.addWidget(self.progress_bar)
        log_group = QGroupBox("Generation Log (User)")
        log_layout_inner = QVBoxLayout()
        self.generation_log = QTextEdit()
        self.generation_log.setReadOnly(True)
        self.generation_log.setFixedHeight(150) # User log fixed height
        self.generation_log.setPlaceholderText("Follow the image generation process here...")
        log_layout_inner.addWidget(self.generation_log)
        log_group.setLayout(log_layout_inner)
        display_log_layout.addWidget(log_group, 1) # Weight 1 for log
        main_splitter.addWidget(display_log_widget)
        main_splitter.setStretchFactor(0, 1) # Controls pane smaller
        main_splitter.setStretchFactor(1, 2) # Display/Log pane larger
        layout.addWidget(main_splitter)

    def _init_system_settings_ui(self):
        layout = QVBoxLayout(self.system_settings_tab)
        form_layout = QFormLayout()
        self.api_key_label = QLabel("Google AI API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow(self.api_key_label, self.api_key_input)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText(f"e.g., {IMAGE_GENERATION_MODEL_CANDIDATE} or gemini-1.5-flash-latest")
        form_layout.addRow("AI Model Name:", self.model_name_input)
        layout.addLayout(form_layout)
        self.save_settings_button = QPushButton("Save Settings & Apply API Key")
        self.save_settings_button.clicked.connect(self.save_and_apply_settings)
        layout.addWidget(self.save_settings_button)
        settings_info = QLabel(
            f"<small><i>Your API key is stored locally. The AI Model specified will be used. "
            f"For real image generation, use an image model name (e.g., '{IMAGE_GENERATION_MODEL_CANDIDATE}') "
            f"and ensure Pillow library is installed. Check Google's documentation for available model names for the Gemini API. "
            f"Otherwise, text models (e.g., 'gemini-1.5-flash-latest') will be used to generate a detailed prompt for a simulated image. "
            f"GCP Project for billing: Your API key is tied to your Google Cloud account; usage will be billed accordingly.</i></small>"
        )
        settings_info.setWordWrap(True)
        layout.addWidget(settings_info)
        layout.addStretch()

    def _init_developer_pipeline_ui(self):
        layout = QVBoxLayout(self.developer_pipeline_tab)
        info_label = QLabel(
            "<h3>Developer & Pipeline Information</h3>"
            "<b>Image Generation (Real):</b><br>"
            "1. <b>User Input:</b> Textual idea/prompt.<br>"
            "2. <b>AI Model (Google Generative AI SDK):</b> Model (e.g., 'imagen-...') uses `genai.GenerativeModel(model_name).generate_content(prompt)`.<br>"
            "3. <b>Response Processing:</b> Image data is extracted from `response.candidates[0].content.parts` where a part has `mime_type` like 'image/png' and data in `inline_data.data`.<br>"
            "4. <b>Display:</b> Image shown via Pillow and QPixmap.<br><br>"
            "<b>Fallback (Simulated Image / Text Model):</b><br>"
            "1. If an image model isn't used, PIL is missing, or real generation fails.<br>"
            "2. A text model (e.g., 'gemini-1.5-flash-latest') refines the user's prompt into a detailed scene description.<br>"
            "3. A placeholder image is generated locally, displaying this scene description.<br><br>"
            "<b>Key API Usage:</b><br>"
            "- `genai.configure(api_key=...)` for global API key setup.<br>"
            "- `genai.GenerativeModel(model_name, generation_config, safety_settings)` for model instantiation.<br>"
            "- `model.generate_content(prompt_string_or_list_of_parts)` for API call.<br>"
            "- Response parsing: `response.candidates`, `candidate.content.parts`, `part.mime_type`, `part.inline_data.data` (for images), `part.text` (for text).<br>"
            "- `response.prompt_feedback` and `candidate.safety_ratings` for content moderation info.<br><br>"
            "<b>Common Issues & Debugging:</b><br>"
            "- Ensure API key is valid and has access to the specified model and the 'generativelanguage.googleapis.com' API.<br>"
            "- Check model name correctness. Model availability can vary by region/project.<br>"
            "- InvalidArgument errors often point to issues with `GenerationConfig` parameters for the specific model, or incorrect prompt structure.<br>"
            "- Monitor the 'Detailed Developer Log' for API request/response clues."
        )
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.RichText) # Allow HTML
        layout.addWidget(info_label)
        dev_log_group = QGroupBox("Detailed Developer Log") # GroupBox for aesthetics
        dev_log_layout_inner = QVBoxLayout()
        self.dev_log_display = QTextEdit()
        self.dev_log_display.setReadOnly(True)
        self.dev_log_display.setPlaceholderText("Developer-level logs, API call details, config objects, worker progress, etc.")
        self.dev_log_display.setFont(QFont("Consolas", 8)) # Monospaced font for logs
        dev_log_layout_inner.addWidget(self.dev_log_display)
        dev_log_group.setLayout(dev_log_layout_inner)
        layout.addWidget(dev_log_group, 1) # Allow dev log to expand

    def load_settings(self):
        loaded_api_key = self.settings.value("api_key", DEFAULT_API_KEY_PLACEHOLDER)
        self.api_key = loaded_api_key # Store the loaded key internally too
        if loaded_api_key == DEFAULT_API_KEY_PLACEHOLDER:
            self.api_key_input.setText("") # Clear if it's placeholder
            self.api_key_input.setPlaceholderText("Enter API Key (currently placeholder)")
        else:
            self.api_key_input.setText(loaded_api_key) # Show actual key (masked)

        default_model_name = IMAGE_GENERATION_MODEL_CANDIDATE if GOOGLE_GENAI_AVAILABLE else 'N/A - AI Library Missing'
        self.model_name_input.setText(self.settings.value("ai_model_name", default_model_name))

        self.img_width_input.setValue(int(self.settings.value("user_img_width", 1024)))
        self.img_height_input.setValue(int(self.settings.value("user_img_height", 1024)))
        self.temperature_input.setValue(float(self.settings.value("user_temperature", 0.9)))
        self.max_tokens_input.setValue(int(self.settings.value("user_max_tokens", 2048)))
        self.prompt_input.setText(self.settings.value("user_last_prompt", ""))
        self.style_input.setText(self.settings.value("user_last_style", "")) # Default to empty

        status_msg = "Settings loaded. "
        if self.api_key and self.api_key != DEFAULT_API_KEY_PLACEHOLDER:
            status_msg += "API Key found. Attempting to configure AI services..."
            self.status_bar.showMessage(status_msg)
            # Configure GenAI using the key from settings (which is now also in self.api_key)
            if not self.configure_genai_globally(self.api_key):
                self.status_bar.showMessage("Failed to configure AI with stored API key. Check settings.", 10000)
            # else: configure_genai_globally will show success message
        else:
            status_msg += "API Key is NOT set or is default. Please configure in System Settings."
            self.status_bar.showMessage(status_msg, 10000) # Longer display for actionable message
        self.update_dev_log("App loaded. " + status_msg)


    def configure_genai_globally(self, key_to_try=None): # Allow passing key directly
        self.genai_configured_successfully = False
        if not GOOGLE_GENAI_AVAILABLE:
            msg = "Google GenAI library not found. Cannot configure."
            self.update_dev_log(msg)
            self.status_bar.showMessage(msg, 10000)
            return False

        if key_to_try is None: # If no key passed, get from input or stored self.api_key
            key_to_try = self.api_key_input.text().strip()
            if not key_to_try or key_to_try == DEFAULT_API_KEY_PLACEHOLDER: # Fallback to stored key if input is empty/placeholder
                 if self.api_key and self.api_key != DEFAULT_API_KEY_PLACEHOLDER:
                    key_to_try = self.api_key
                 else: # No valid key available from input or stored
                    key_to_try = None


        if key_to_try and key_to_try != DEFAULT_API_KEY_PLACEHOLDER:
            try:
                self.update_dev_log(f"Configuring genai with API key: {'*' * (len(key_to_try) - 4) + key_to_try[-4:]}")
                genai.configure(api_key=key_to_try)
                
                # Test model accessibility
                test_model_name = self.model_name_input.text().strip()
                if not test_model_name: # If model name input is empty, use candidate
                    test_model_name = IMAGE_GENERATION_MODEL_CANDIDATE
                
                self.update_dev_log(f"Attempting to get model: '{test_model_name}' to verify API key and model access.")
                genai.get_model(test_model_name) # This will raise if model not found or key invalid for it
                
                msg = f"Google GenAI configured successfully with API key. Model '{test_model_name}' accessible."
                self.update_dev_log(msg)
                self.status_bar.showMessage(msg, 7000)
                self.api_key = key_to_try # Update internal API key state
                self.genai_configured_successfully = True
                return True
            except Exception as e:
                msg = f"Error configuring/testing Google GenAI: {type(e).__name__}: {e}"
                QMessageBox.warning(self, "API Key/Model Error", f"{msg}\n\n- Check your API key's validity.\n- Ensure the model name ('{test_model_name}') is correct and available for your key/region.\n- Verify network connectivity to Google APIs.\n- Ensure the 'Generative Language API' (generativelanguage.googleapis.com) is enabled for your GCP project.")
                self.status_bar.showMessage(f"GenAI Config Error: {type(e).__name__}", 10000)
                self.update_dev_log(msg)
                # import traceback # For more detailed error in dev log
                # self.update_dev_log(f"Traceback: {traceback.format_exc()}")
                return False
        else:
            msg = "API key is not set or is a placeholder. GenAI not configured."
            self.update_dev_log(msg)
            self.status_bar.showMessage(msg, 7000)
            return False

    def save_and_apply_settings(self):
        new_api_key_from_input = self.api_key_input.text().strip()
        current_model_name = self.model_name_input.text().strip()

        key_to_save_and_use = new_api_key_from_input
        if not new_api_key_from_input: # If input is empty, revert to trying stored self.api_key, or placeholder
            if self.api_key and self.api_key != DEFAULT_API_KEY_PLACEHOLDER:
                key_to_save_and_use = self.api_key # Use previously valid key if input is cleared
            else:
                key_to_save_and_use = DEFAULT_API_KEY_PLACEHOLDER
        
        self.settings.setValue("api_key", key_to_save_and_use) # Save even if it's placeholder
        self.settings.setValue("ai_model_name", current_model_name)
        self.settings.setValue("user_img_width", self.img_width_input.value())
        self.settings.setValue("user_img_height", self.img_height_input.value())
        self.settings.setValue("user_temperature", self.temperature_input.value())
        self.settings.setValue("user_max_tokens", self.max_tokens_input.value())
        self.update_dev_log("Settings saved to local QSettings.")

        # Re-configure with the (potentially new) key from input or saved valid key
        config_success = self.configure_genai_globally(key_to_save_and_use)

        if config_success:
            QMessageBox.information(self, "Settings Applied", "Settings saved and Google AI API Key applied and verified.")
            self.status_bar.showMessage("Settings saved and AI configured.", 5000)
        else:
            # configure_genai_globally already showed a QMessageBox for errors
            self.status_bar.showMessage("Settings saved, but AI configuration failed or key is invalid/missing.", 7000)


    def update_generation_log(self, message):
        self.generation_log.append(message)
        QApplication.processEvents() # Keep UI responsive during logging

    def update_dev_log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.dev_log_display.append(f"[{timestamp}] {message}")
        QApplication.processEvents() # Keep UI responsive

    def start_image_generation(self):
        prompt_base = self.prompt_input.toPlainText().strip()
        style_suffix = self.style_input.text().strip()

        if not prompt_base:
            QMessageBox.warning(self, "Input Error", "Please enter an image idea/prompt.")
            return

        final_prompt = prompt_base
        if style_suffix:
            final_prompt = f"{prompt_base}, style: {style_suffix}"
        
        self.update_generation_log(f"User Prompt (with style): '{final_prompt[:150]}...'")


        if not GOOGLE_GENAI_AVAILABLE:
            self.update_generation_log("Google AI library not available. Simulating.")
            self._handle_simulation_fallback("AI Library Missing")
            return

        if not self.genai_configured_successfully:
            QMessageBox.warning(self, "API Key/Configuration Error", "Google AI is not configured successfully. Please check API Key and Model Name in System Settings.")
            self.tabs.setCurrentWidget(self.system_settings_tab)
            return

        current_model_name = self.model_name_input.text().strip()
        if not current_model_name:
            QMessageBox.warning(self, "Configuration Error", "AI Model Name is not set in System Settings.")
            self.tabs.setCurrentWidget(self.system_settings_tab)
            return

        is_intended_image_model = "imagen" in current_model_name or "image" in current_model_name # Simple check

        if is_intended_image_model and not PIL_AVAILABLE:
            QMessageBox.warning(self, "Library Missing", "Pillow (PIL) library is needed for real image generation with the selected model but not found. Install: pip install Pillow. Will attempt text-based simulation as fallback.")
            # Fallback will be handled by worker logic if attempt_real_image is true but PIL is false

        self.generate_button.setEnabled(False)
        self.save_image_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate progress
        self.image_display_label.setText("🧠 AI is thinking... Please wait.")
        self.image_display_label.setPixmap(QPixmap()) # Clear previous image
        # self.generation_log.clear() # Cleared by update_generation_log if first message
        self.update_generation_log(f"Starting generation for: '{final_prompt[:100]}...'") # User log
        self.update_dev_log(f"User prompt for worker: '{final_prompt}'")
        self.update_dev_log(f"Using model: {current_model_name}")

        generation_params = {
            "model_name": current_model_name,
            "temperature": self.temperature_input.value(),
            "max_output_tokens": self.max_tokens_input.value(), # For text fallback
            "top_p": self.temperature_input.value() + 0.1 if self.temperature_input.value() < 0.9 else 1.0, # Example logic for Top P
            "top_k": 40 if self.temperature_input.value() > 0.5 else None, # Example logic for Top K
            "target_image_width": self.img_width_input.value(), # For simulation display
            "target_image_height": self.img_height_input.value(), # For simulation display
            # "style" is already incorporated into final_prompt
        }
        self.update_dev_log(f"Generation parameters for worker: {generation_params}")

        # Worker decides if it can do real image based on model name and PIL
        self.worker = ImageGenerationWorker(self.api_key, final_prompt, generation_params, attempt_real_image=True)
        
        # Connect signals: User log gets less verbose, Dev log gets all progress.
        # self.worker.progress.connect(self.update_generation_log) # Can make user log too noisy
        self.worker.progress.connect(lambda msg: self.update_dev_log(f"[Worker] {msg}")) # All worker progress to dev log
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.start()

    def _handle_simulation_fallback(self, reason_title="Simulation Fallback"):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1) # Reset progress bar
        self.generate_button.setEnabled(True)
        self.image_display_label.setText(f"{reason_title}\nDisplaying a placeholder.")
        
        # Create a more informative placeholder
        dummy_image = QImage(self.img_width_input.value(), self.img_height_input.value(), QImage.Format_RGB32)
        dummy_image.fill(Qt.darkGray)
        painter = QPainter(dummy_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(Qt.white))
        
        font_title = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font_title)
        text_rect_title = QRect(dummy_image.rect().adjusted(10, 10, -10, -dummy_image.height() * 2 // 3))
        painter.drawText(text_rect_title, Qt.AlignCenter | Qt.TextWordWrap, reason_title)

        font_prompt = QFont("Arial", 8)
        painter.setFont(font_prompt)
        prompt_text = self.prompt_input.toPlainText()
        display_prompt = f"Original Idea: '{prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}'"
        text_rect_prompt = QRect(dummy_image.rect().adjusted(10, dummy_image.height() // 3, -10, -10))
        painter.drawText(text_rect_prompt, Qt.AlignCenter | Qt.TextWordWrap, display_prompt)
        painter.end()
        
        self.current_pixmap = QPixmap.fromImage(dummy_image)
        self._display_scaled_pixmap()
        self.save_image_button.setEnabled(True) # Allow saving placeholder
        self.update_generation_log(f"{reason_title}. Switched to basic placeholder for '{self.prompt_input.toPlainText()[:50]}...'.")


    def on_generation_finished(self, result):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1) # Reset progress bar
        self.generate_button.setEnabled(True)

        if isinstance(result, QPixmap):
            self.current_pixmap = result
            if self.current_pixmap.isNull():
                self.update_generation_log("Generation finished, but received a null QPixmap from worker.")
                self.update_dev_log("Worker returned null QPixmap.")
                self._handle_simulation_fallback("Error: Received Null Image")
            else:
                self._display_scaled_pixmap()
                self.save_image_button.setEnabled(True)
                self.update_generation_log("Image generation successful!")
                self.status_bar.showMessage("Image ready!", 5000)
        elif isinstance(result, str) and result.startswith("Error:"):
            error_message = result
            self.image_display_label.setText(f"Generation Failed.\n{error_message.split(':', 1)[1].strip()}") # Show cleaner error
            self.update_generation_log(f"Generation failed: {error_message}") # Full error to log
            self.update_dev_log(f"Worker finished with error string: {result}")
            self.current_pixmap = None # No valid image
            concise_error = error_message.split(':', 1)[1][:100].strip() if ':' in error_message else error_message[:100]
            self._handle_simulation_fallback(f"Error: {concise_error}...")
        else: # Unexpected result
            self.image_display_label.setText("Generation finished with an unexpected result type.")
            self.update_generation_log(f"Generation ended with unexpected result type: {type(result)}")
            self.update_dev_log(f"Unexpected result object from worker: {str(result)[:200]}")
            self.current_pixmap = None
            self._handle_simulation_fallback("Unexpected Result Type")

    def _display_scaled_pixmap(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            label_w = self.image_display_label.width()
            label_h = self.image_display_label.height()
            if label_w <= 10 or label_h <= 10: # Avoid issues if label isn't sized yet
                self.image_display_label.setPixmap(self.current_pixmap) # Show unscaled if label is tiny
                return
            
            # Scale pixmap to fit label while keeping aspect ratio
            scaled_pixmap = self.current_pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display_label.setPixmap(scaled_pixmap)
        else: # No valid current_pixmap
            if not self.generate_button.isEnabled(): # If generating
                self.image_display_label.setText("🧠 AI is thinking... Please wait.")
            else: # Idle or error
                self.image_display_label.setText("Image will appear here, or an error occurred.")


    def save_current_image(self):
        if not self.current_pixmap or self.current_pixmap.isNull():
            QMessageBox.warning(self, "Save Error", "No valid image to save.")
            return

        # Create a filename from the prompt
        prompt_text_for_filename = self.prompt_input.toPlainText()[:30].strip()
        # Sanitize filename: replace non-alphanumeric with underscore
        sanitized_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt_text_for_filename)
        default_filename = f"gemini_art_{sanitized_prompt}.png" if sanitized_prompt else "gemini_generated_image.png"
        
        # Default save directory
        save_dir = os.path.join(os.path.expanduser("~"), "Pictures", "GeminiSmartAppImages") # More specific folder
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except OSError: # Fallback if Pictures/AppFolder fails
                self.update_dev_log(f"Could not create save directory {save_dir}, falling back to user's home.")
                save_dir = os.path.join(os.path.expanduser("~")) 

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", 
                                                 os.path.join(save_dir, default_filename),
                                                 "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)")
        if filePath:
            if self.current_pixmap.save(filePath):
                self.status_bar.showMessage(f"Image saved to {filePath}", 5000)
                self.update_generation_log(f"Image successfully saved: {filePath}")
            else:
                QMessageBox.critical(self, "Save Error", f"Could not save image to {filePath}. Check permissions or disk space.")
                self.update_generation_log(f"Failed to save image to {filePath}.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'image_display_label') and self.image_display_label.isVisible():
            # Call display_scaled_pixmap only if an image is actually loaded
            if self.current_pixmap and not self.current_pixmap.isNull():
                 self._display_scaled_pixmap()

    def closeEvent(self, event):
        self.update_dev_log("Close event triggered. Saving settings...")
        self.settings.setValue("user_last_prompt", self.prompt_input.toPlainText())
        self.settings.setValue("user_last_style", self.style_input.text())
        # API key and model name are saved when "Save Settings" is clicked, not necessarily on close.
        # Or, we can save them here too:
        # self.settings.setValue("api_key", self.api_key_input.text().strip() or DEFAULT_API_KEY_PLACEHOLDER)
        # self.settings.setValue("ai_model_name", self.model_name_input.text().strip())
        self.settings.sync() # Ensure settings are written to disk
        self.update_dev_log("Settings synced.")

        if self.worker and self.worker.isRunning():
            self.update_dev_log("Attempting to stop running worker thread before closing...")
            self.worker.quit() # Request clean exit
            if not self.worker.wait(1500): # Wait 1.5 seconds
                self.update_dev_log("Worker thread did not stop gracefully. Terminating.")
                self.worker.terminate() # Force stop
                self.worker.wait() # Wait for termination
            else:
                self.update_dev_log("Worker thread stopped successfully.")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Apply a modern style if available
    if "Fusion" in QApplication.style().objectName() or "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    
    main_window = SmartImageApp()
    main_window.show()
    sys.exit(app.exec_())
