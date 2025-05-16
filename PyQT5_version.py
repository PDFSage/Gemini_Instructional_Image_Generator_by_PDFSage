import sys
import json # Not used in this version, QSettings handles it
import os
import time # For simulation

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget, QFileDialog,
    QGraphicsView, QGraphicsScene, QProgressBar, QMessageBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QSplitter, QGroupBox # Added QSplitter and QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QRect # Added QRect

# Attempt to import google.generativeai, but make it optional for UI testing
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    print("WARNING: google.generativeai library not found. AI features will be disabled.")

# --- Configuration ---
DEFAULT_API_KEY = "YOUR_GOOGLE_AI_API_KEY_HERE" # Important: User must replace this

# --- Image Generation Worker (Text Prompt Enhancement + Simulated Image) ---
class ImageGenerationWorker(QThread):
    finished = pyqtSignal(object) # QPixmap for success, str for error message
    progress = pyqtSignal(str)    # For sending log messages to UI

    def __init__(self, api_key, user_prompt, generation_params):
        super().__init__()
        self.api_key = api_key
        self.user_prompt = user_prompt
        self.generation_params = generation_params
        self.client = None # Not directly used if genai.configure is global

    def run(self):
        if not GOOGLE_GENAI_AVAILABLE:
            self.progress.emit("Google GenAI library not available.")
            self.progress.emit("Simulating image generation (AI library missing)...")
            time.sleep(1) # Simulate work
            # Create a dummy image reflecting the lack of AI
            image = QImage(self.generation_params.get("target_image_width", 512),
                           self.generation_params.get("target_image_height", 512),
                           QImage.Format_RGB32)
            image.fill(Qt.lightGray)
            painter = QPainter(image)
            painter.setPen(QColor(Qt.black))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(image.rect(), Qt.AlignCenter, f"Simulated Image (AI Offline)\nPrompt: '{self.user_prompt[:50]}...'")
            painter.end()
            self.finished.emit(QPixmap.fromImage(image))
            return

        if not self.api_key or self.api_key == DEFAULT_API_KEY or self.api_key == "YOUR_GOOGLE_AI_API_KEY_HERE":
            self.progress.emit("API Key not configured or is placeholder.")
            self.finished.emit("Error: API Key missing or invalid. Please set it in System Settings.")
            return

        try:
            self.progress.emit("Configuring AI model with provided API key...")
            # The API key should be configured once globally, or per model instance.
            # genai.configure(api_key=self.api_key) # Usually done once at app start or when key changes

            model_name = self.generation_params.get("model_name", 'gemini-1.5-flash-latest')
            
            gen_config_dict = {
                "temperature": self.generation_params.get("temperature", 0.7),
                "top_p": self.generation_params.get("top_p", 0.95), # Example, can be None
                "top_k": self.generation_params.get("top_k", 40),   # Example, can be None
                "max_output_tokens": self.generation_params.get("max_output_tokens", 300),
            }
            # Filter out None values if any param is not set or to use model defaults
            generation_config = {k: v for k, v in gen_config_dict.items() if v is not None}

            safety_settings = [ # Example safety settings
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
                # system_instruction can be powerful here.
            )
            
            self.progress.emit(f"Generating detailed description for image prompt: '{self.user_prompt}' using {model_name}...")
            
            # Construct a prompt for the LLM to generate a detailed image description
            # This is the "instructional" part for the image generation
            text_prompt_for_llm = (
                f"You are an AI assistant that generates detailed, vivid, and creative "
                f"scene descriptions for an AI image generator. Based on the user's idea: "
                f"'{self.user_prompt}', create a rich description. Focus on visual elements, "
                f"atmosphere, style (e.g., photorealistic, fantasy, cartoon), composition, "
                f"lighting, and any specific objects or characters mentioned. "
                f"The output should be a paragraph of descriptive text."
            )
            
            response = model.generate_content(text_prompt_for_llm)
            
            if not response.candidates or not response.text:
                # Check for blocked prompt or other issues
                try:
                    error_info = str(response.prompt_feedback)
                except Exception:
                    error_info = "No specific feedback available."
                self.progress.emit(f"No content generated by the model. Feedback: {error_info}")
                self.finished.emit(f"Error: AI model returned no content. {error_info}")
                return

            detailed_description = response.text
            self.progress.emit(f"AI Generated Detailed Description (first 100 chars): {detailed_description[:100]}...")
            self.progress.emit(f"Full AI Description:\n{detailed_description}")
            
            # --- Placeholder for actual image generation ---
            self.progress.emit("Simulating actual image generation using the detailed description...")
            time.sleep(0.5) # Simulate network latency for image model

            img_width = self.generation_params.get("target_image_width", 512)
            img_height = self.generation_params.get("target_image_height", 512)

            image = QImage(img_width, img_height, QImage.Format_RGB32)
            image.fill(QColor("#2E2E2E")) # Dark background
            
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Title
            painter.setPen(QColor(Qt.white))
            title_font = QFont("Arial", 14, QFont.Bold)
            painter.setFont(title_font)
            painter.drawText(QRect(10, 10, img_width - 20, 40), Qt.AlignHCenter | Qt.AlignTop, f"Instructional Image (Simulated)")

            # User Prompt
            painter.setPen(QColor("#AAAAAA"))
            prompt_font = QFont("Arial", 9, QFont.StyleItalic)
            painter.setFont(prompt_font)
            user_prompt_rect = QRect(15, 50, img_width - 30, 40)
            painter.drawText(user_prompt_rect, Qt.AlignLeft | Qt.TextWordWrap, f"User Idea: {self.user_prompt}")
            
            # AI Generated Description
            painter.setPen(QColor("#DDDDDD"))
            desc_font = QFont("Arial", 8)
            painter.setFont(desc_font)
            description_rect = QRect(15, 100, img_width - 30, img_height - 110)
            painter.drawText(description_rect, Qt.AlignLeft | Qt.TextWordWrap, f"AI Generated Detailed Description:\n{detailed_description}")
            
            painter.end()
            final_pixmap = QPixmap.fromImage(image)
            self.finished.emit(final_pixmap)

        except Exception as e:
            self.progress.emit(f"Error during AI processing: {type(e).__name__}: {e}")
            self.finished.emit(f"Error: {type(e).__name__}: {e}")


# --- Main Application Window ---
class SmartImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Instructional Image Generation Tool")
        self.setGeometry(50, 50, 900, 700) # Adjusted size

        # Using QSettings for persistence
        self.settings = QSettings("MyCompany", "SmartImageApp")
        self.api_key = "" # Loaded from settings

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Create tabs
        self.user_request_tab = QWidget()
        self.system_settings_tab = QWidget()
        self.developer_pipeline_tab = QWidget()

        self.tabs.addTab(self.user_request_tab, "🎨 User Image Generation")
        self.tabs.addTab(self.system_settings_tab, "⚙️ System Settings")
        self.tabs.addTab(self.developer_pipeline_tab, "🛠️ Developer & Pipeline")

        self._init_user_request_ui()
        self._init_system_settings_ui()
        self._init_developer_pipeline_ui()

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Application Ready. Please configure API Key in System Settings if not done.")

        self.current_pixmap = None # To store the last generated pixmap

        self.load_settings() # Load settings, which also configures genai if key exists


    def _init_user_request_ui(self):
        layout = QVBoxLayout(self.user_request_tab)
        main_splitter = QSplitter(Qt.Horizontal)

        # Left side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Prompt input
        self.prompt_label = QLabel("<b>Enter Your Image Idea:</b>")
        controls_layout.addWidget(self.prompt_label)
        self.prompt_input = QTextEdit() # Changed to QTextEdit for multi-line prompts
        self.prompt_input.setPlaceholderText("e.g., 'A mystical forest with glowing mushrooms and a hidden path, photorealistic sunset lighting'")
        self.prompt_input.setFixedHeight(80)
        controls_layout.addWidget(self.prompt_input)

        # Generation Parameters
        params_group = QGroupBox("Generation Parameters")
        params_form_layout = QFormLayout()

        self.img_width_input = QSpinBox()
        self.img_width_input.setRange(256, 4096); self.img_width_input.setValue(512); self.img_width_input.setSuffix(" px")
        params_form_layout.addRow("Simulated Image Width:", self.img_width_input)

        self.img_height_input = QSpinBox()
        self.img_height_input.setRange(256, 4096); self.img_height_input.setValue(512); self.img_height_input.setSuffix(" px")
        params_form_layout.addRow("Simulated Image Height:", self.img_height_input)
        
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 2.0); self.temperature_input.setValue(0.8); self.temperature_input.setSingleStep(0.05)
        params_form_layout.addRow("Text Gen Temperature:", self.temperature_input)
        
        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setRange(50, 8000); self.max_tokens_input.setValue(300) # Increased for detailed descriptions
        params_form_layout.addRow("Text Gen Max Tokens:", self.max_tokens_input)
        
        # Add more genai specific params later if needed (top_k, top_p)
        params_group.setLayout(params_form_layout)
        controls_layout.addWidget(params_group)

        # Action Buttons
        self.generate_button = QPushButton("✨ Generate Image")
        self.generate_button.setFixedHeight(40)
        self.generate_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 16px; border-radius: 5px; } QPushButton:hover { background-color: #45a049; }")
        self.generate_button.clicked.connect(self.start_image_generation)
        controls_layout.addWidget(self.generate_button)

        self.alter_button = QPushButton("🔄 Alter Image (Conceptual)")
        self.alter_button.setEnabled(False) 
        self.alter_button.clicked.connect(self.conceptual_alter_image)
        controls_layout.addWidget(self.alter_button)

        self.save_image_button = QPushButton("💾 Save Image")
        self.save_image_button.setEnabled(False)
        self.save_image_button.clicked.connect(self.save_current_image)
        controls_layout.addWidget(self.save_image_button)
        
        controls_layout.addStretch() # Pushes everything up
        main_splitter.addWidget(controls_widget)

        # Right side: Image Display and Log
        display_log_widget = QWidget()
        display_log_layout = QVBoxLayout(display_log_widget)

        self.image_display_label = QLabel("Your generated image will appear here.")
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setMinimumSize(400, 300) # Initial minimum
        self.image_display_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f8f8f8; border-radius: 5px;")
        display_log_layout.addWidget(self.image_display_label, 3) # Give more stretch factor

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        display_log_layout.addWidget(self.progress_bar)
        
        log_group = QGroupBox("Generation Log")
        log_layout = QVBoxLayout()
        self.generation_log = QTextEdit()
        self.generation_log.setReadOnly(True)
        self.generation_log.setFixedHeight(150) # Fixed height for log
        self.generation_log.setPlaceholderText("Follow the image generation process here...")
        log_layout.addWidget(self.generation_log)
        log_group.setLayout(log_layout)
        display_log_layout.addWidget(log_group, 1) # Less stretch factor

        main_splitter.addWidget(display_log_widget)
        main_splitter.setStretchFactor(0, 1) # Controls side less stretchy
        main_splitter.setStretchFactor(1, 2) # Display side more stretchy
        layout.addWidget(main_splitter)


    def _init_system_settings_ui(self):
        layout = QVBoxLayout(self.system_settings_tab)
        form_layout = QFormLayout()

        self.api_key_label = QLabel("Google AI API Key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter your Gemini API Key")
        form_layout.addRow(self.api_key_label, self.api_key_input)

        self.model_name_input = QLineEdit() # Default text model for prompt enhancement
        form_layout.addRow("Text Gen Model Name (for prompt detailing):", self.model_name_input)
        
        layout.addLayout(form_layout)

        self.save_settings_button = QPushButton("Save Settings & Apply API Key")
        self.save_settings_button.clicked.connect(self.save_and_apply_settings)
        layout.addWidget(self.save_settings_button)
        
        settings_info = QLabel(
            "<small><i>Your API key is stored locally using QSettings. "
            "The Text Gen Model is used to refine your prompt before the (simulated) image generation. "
            "Ensure you use a valid model name like 'gemini-1.5-flash-latest' or 'gemini-pro'.</i></small>"
        )
        settings_info.setWordWrap(True)
        layout.addWidget(settings_info)
        layout.addStretch()


    def _init_developer_pipeline_ui(self):
        layout = QVBoxLayout(self.developer_pipeline_tab)
        info_label = QLabel(
            "<h3>Developer & Pipeline Information</h3>"
            "This section outlines the conceptual data flow and offers tools for developers.<br><br>"
            "<b>Current Instructional Image Generation Pipeline:</b><br>"
            "1. <b>User Input:</b> User provides a textual idea/prompt for an image.<br>"
            "2. <b>Prompt Enhancement (AI - Text Model):</b> The user's input is sent to a Google Generative AI text model (e.g., Gemini). "
            "This model expands, refines, and details the initial idea into a rich textual description suitable for an image generator.<br>"
            "   <i>(This step uses the configured API Key and Text Gen Model)</i><br>"
            "3. <b>Image Generation (Simulated):</b> The detailed textual description from step 2 would ideally be fed into a dedicated AI image generation model (e.g., Imagen). "
            "   <i><b>In this application, this step is SIMULATED.</b> A placeholder image is created displaying the user prompt and the AI-enhanced description.</i><br>"
            "4. <b>Image Display:</b> The final (simulated) image is shown to the user.<br><br>"
            "<b>Developer Tools:</b>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.dev_log_display = QTextEdit()
        self.dev_log_display.setReadOnly(True)
        self.dev_log_display.setPlaceholderText("Developer-level logs, API call details, and pipeline step outputs would appear here.")
        layout.addWidget(self.dev_log_display, 1) # Make it expandable
        
        self.test_text_gen_button = QPushButton("Test Text Prompt Refinement Stage")
        self.test_text_gen_button.setToolTip("Uses the current prompt from User Tab to test only the text generation stage.")
        self.test_text_gen_button.clicked.connect(self.run_text_refinement_test)
        layout.addWidget(self.test_text_gen_button)


    def load_settings(self):
        self.api_key = self.settings.value("api_key", DEFAULT_API_KEY)
        self.api_key_input.setText(self.api_key)
        
        default_model = 'gemini-1.5-flash-latest' if GOOGLE_GENAI_AVAILABLE else 'N/A - AI Library Missing'
        self.model_name_input.setText(self.settings.value("text_model_name", default_model))

        self.img_width_input.setValue(int(self.settings.value("user_img_width", 512)))
        self.img_height_input.setValue(int(self.settings.value("user_img_height", 512)))
        self.temperature_input.setValue(float(self.settings.value("user_temperature", 0.8)))
        self.max_tokens_input.setValue(int(self.settings.value("user_max_tokens", 300)))
        self.prompt_input.setText(self.settings.value("user_last_prompt", ""))

        status_msg = "Settings loaded. "
        if self.api_key and self.api_key != DEFAULT_API_KEY:
            status_msg += f"API Key is set. Attempting to configure AI services..."
            self.configure_genai_globally() # Attempt to configure on load
        else:
            status_msg += "API Key is NOT set or is default. Please configure in System Settings."
        
        self.status_bar.showMessage(status_msg)
        self.update_dev_log(status_msg)

    def configure_genai_globally(self):
        if GOOGLE_GENAI_AVAILABLE and self.api_key and self.api_key != DEFAULT_API_KEY:
            try:
                genai.configure(api_key=self.api_key)
                msg = "Google GenAI configured successfully with the API key."
                self.status_bar.showMessage(msg)
                self.update_dev_log(msg)
                return True
            except Exception as e:
                msg = f"Error configuring Google GenAI: {e}"
                QMessageBox.warning(self, "API Key Error", f"Failed to configure Google GenAI with the API key: {e}\nCheck your API key and network connection.")
                self.status_bar.showMessage(msg, 10000) # Show for 10s
                self.update_dev_log(msg)
                return False
        elif not GOOGLE_GENAI_AVAILABLE:
            msg = "Google GenAI library not found. Cannot configure."
            self.update_dev_log(msg)
            return False
        else:
            msg = "API key is not set or is a placeholder. GenAI not configured."
            self.update_dev_log(msg)
            return False


    def save_and_apply_settings(self):
        old_api_key = self.api_key
        self.api_key = self.api_key_input.text()
        
        self.settings.setValue("api_key", self.api_key)
        self.settings.setValue("text_model_name", self.model_name_input.text())

        self.settings.setValue("user_img_width", self.img_width_input.value()) # Save user UI defaults too
        self.settings.setValue("user_img_height", self.img_height_input.value())
        self.settings.setValue("user_temperature", self.temperature_input.value())
        self.settings.setValue("user_max_tokens", self.max_tokens_input.value())
        self.settings.setValue("user_last_prompt", self.prompt_input.toPlainText())

        self.update_dev_log("Settings saved to persistent storage.")
        
        if self.api_key != old_api_key or not genai.API_KEY: # If key changed or was never set
            if self.configure_genai_globally():
                 QMessageBox.information(self, "Settings Saved", "Settings saved and Google AI API Key applied successfully.")
            # Error messages handled by configure_genai_globally
        else:
            QMessageBox.information(self, "Settings Saved", "Settings saved. API key was unchanged or already configured.")
        self.status_bar.showMessage("Settings saved.", 5000)


    def update_generation_log(self, message):
        self.generation_log.append(message)
        self.dev_log_display.append(f"[User Log] {message}") # Mirror to dev log

    def update_dev_log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.dev_log_display.append(f"[{timestamp}] {message}")

    def start_image_generation(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Input Error", "Please enter an image idea/prompt.")
            return

        if not GOOGLE_GENAI_AVAILABLE:
            QMessageBox.warning(self, "Setup Error", "Google Generative AI library is not installed. AI features are disabled.")
            # Fallback to very basic simulation if user still clicks generate
            self.update_generation_log("AI library missing, running offline simulation.")
            self.on_generation_finished(None) # Trigger simulation without AI
            return

        if not self.api_key or self.api_key == DEFAULT_API_KEY or not genai.API_KEY: # Check if genai is truly configured
            QMessageBox.warning(self, "API Key Error", "Google AI API Key is not configured or is invalid. Please set it in System Settings and click 'Save & Apply'.")
            self.tabs.setCurrentWidget(self.system_settings_tab)
            return

        self.generate_button.setEnabled(False)
        self.alter_button.setEnabled(False)
        self.save_image_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0,0) # Indeterminate progress
        self.image_display_label.setText("🧠 Thinking and generating detailed prompt...")
        self.image_display_label.setPixmap(QPixmap()) # Clear previous image
        self.generation_log.clear()
        self.update_generation_log(f"Starting generation for prompt: '{prompt}'")

        generation_params = {
            "model_name": self.model_name_input.text(),
            "temperature": self.temperature_input.value(),
            "max_output_tokens": self.max_tokens_input.value(),
            "target_image_width": self.img_width_input.value(),
            "target_image_height": self.img_height_input.value(),
            # Add top_p, top_k from UI if you add those controls
        }

        self.worker = ImageGenerationWorker(self.api_key, prompt, generation_params)
        self.worker.progress.connect(self.update_generation_log)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.start()

    def on_generation_finished(self, result):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0,1) # Reset
        self.generate_button.setEnabled(True)

        if isinstance(result, QPixmap):
            self.current_pixmap = result # Store original full-res pixmap
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = self.current_pixmap.scaled(self.image_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display_label.setPixmap(scaled_pixmap)
            self.alter_button.setEnabled(True)
            self.save_image_button.setEnabled(True)
            self.update_generation_log("Image generation (simulated) successful.")
        elif isinstance(result, str) and result.startswith("Error:"):
            self.image_display_label.setText(f"Generation Failed.\n{result}")
            QMessageBox.critical(self, "Generation Error", result)
            self.update_generation_log(f"Generation failed: {result}")
            self.current_pixmap = None
        elif result is None and not GOOGLE_GENAI_AVAILABLE: # Offline simulation path
             # Create a dummy image for offline simulation
            dummy_image = QImage(self.img_width_input.value(), self.img_height_input.value(), QImage.Format_RGB32)
            dummy_image.fill(Qt.darkGray)
            painter = QPainter(dummy_image)
            painter.setPen(QColor(Qt.white))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(dummy_image.rect(), Qt.AlignCenter, f"Offline Simulated Image\nNo AI Library Found.\nPrompt: '{self.prompt_input.toPlainText()[:50]}...'")
            painter.end()
            self.current_pixmap = QPixmap.fromImage(dummy_image)
            scaled_pixmap = self.current_pixmap.scaled(self.image_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display_label.setPixmap(scaled_pixmap)
            self.update_generation_log("Offline simulation complete (AI library missing).")
            self.save_image_button.setEnabled(True) # Can still save the dummy image
        else:
            self.image_display_label.setText("Generation finished with an unexpected result or was cancelled.")
            self.update_generation_log(f"Generation ended with result: {result}")
            self.current_pixmap = None
            
    def conceptual_alter_image(self):
        if not self.current_pixmap:
            QMessageBox.information(self, "Alter Image", "No image available to alter.")
            return
        
        # This is purely conceptual. A real implementation would involve:
        # 1. UI for alteration type (inpainting, style change, variation).
        # 2. Sending the original image (or its ID/seed) and new instructions to an image editing AI.
        QMessageBox.information(self, "Conceptual Alteration", 
                                "This would be where you'd implement image alteration features.\n"
                                "For example, you could take the current 'AI Generated Detailed Description' (from logs), "
                                "allow the user to modify it, and then re-run the (simulated) generation.")
        self.tabs.setCurrentWidget(self.developer_pipeline_tab) # Show dev log as it might have the detailed prompt
        self.generation_log.append("\n--- User initiated 'Alter Image' (Conceptual) ---\n"
                                   "Review the 'AI Generated Detailed Description' in the logs. "
                                   "Modify it and paste into a new prompt for a variation.")


    def save_current_image(self):
        if not self.current_pixmap:
            QMessageBox.warning(self, "Save Error", "No image to save.")
            return

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.expanduser("~/generated_image.png"),
                                                  "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)")
        if filePath:
            if self.current_pixmap.save(filePath):
                self.status_bar.showMessage(f"Image saved to {filePath}", 5000)
                self.update_generation_log(f"Image saved to: {filePath}")
            else:
                QMessageBox.critical(self, "Save Error", f"Could not save image to {filePath}.")
                self.update_generation_log(f"Failed to save image to: {filePath}")
            
    def run_text_refinement_test(self):
        test_prompt = self.prompt_input.toPlainText().strip()
        if not test_prompt:
            test_prompt = "A friendly robot offering a flower, Pixar style" # Default for test
        
        self.update_dev_log(f"\n--- Developer Test: Text Prompt Refinement Stage ---")
        self.update_dev_log(f"Input Prompt for Test: '{test_prompt}'")

        if not GOOGLE_GENAI_AVAILABLE:
            self.update_dev_log("Google GenAI library not available. Test cannot run.")
            QMessageBox.warning(self, "Test Error", "Google GenAI library not available.")
            return

        if not self.api_key or self.api_key == DEFAULT_API_KEY or not genai.API_KEY:
            self.update_dev_log("API Key not configured or GenAI not initialized. Test cannot run.")
            QMessageBox.warning(self, "API Key Error", "Please configure API Key in System Settings and Save/Apply.")
            return
        
        try:
            # genai.configure(api_key=self.api_key) # Should be configured globally
            model_name = self.model_name_input.text() or 'gemini-1.5-flash-latest'
            model = genai.GenerativeModel(model_name) # Use default config for test
            
            text_prompt_for_llm = (
                f"You are an AI assistant that generates detailed, vivid, and creative "
                f"scene descriptions for an AI image generator. Based on the user's idea: "
                f"'{test_prompt}', create a rich description. Focus on visual elements, "
                f"atmosphere, style (e.g., photorealistic, fantasy, cartoon), composition, "
                f"lighting, and any specific objects or characters mentioned. "
                f"The output should be a paragraph of descriptive text."
            )
            
            self.update_dev_log(f"Sending to LLM ({model.model_name}): {text_prompt_for_llm[:150]}...")
            response = model.generate_content(text_prompt_for_llm)
            
            if response.text:
                self.update_dev_log(f"LLM Response (Refined Prompt):\n{response.text}")
            else:
                self.update_dev_log(f"LLM generated no text. Feedback: {response.prompt_feedback}")
        except Exception as e:
            self.update_dev_log(f"Error during text refinement test: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "Test Error", f"Error during text refinement: {e}")
        self.update_dev_log(f"--- End of Developer Text Refinement Test ---")

    # Ensure image display resizes when window resizes
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(self.image_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_display_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        # Save settings on close (QSettings often auto-saves, but explicit can be good)
        self.settings.setValue("user_last_prompt", self.prompt_input.toPlainText()) # Save last prompt
        self.settings.sync() # Ensure data is written
        self.update_dev_log("Application closing, settings synced.")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Apply a basic style
    app.setStyle("Fusion") 
    # Or a more modern stylesheet could be loaded here
    # Example:
    # app.setStyleSheet("""
    #     QMainWindow { background-color: #f0f0f0; }
    #     QPushButton { padding: 5px; }
    #     QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox { padding: 3px; border: 1px solid #ccc; border-radius: 3px; }
    #     QGroupBox { font-weight: bold; margin-top: 10px; }
    #     QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
    # """)

    main_window = SmartImageApp()
    main_window.show()
    sys.exit(app.exec_())