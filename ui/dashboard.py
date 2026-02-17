"""
Green-Inference: UI Dashboard
Modern dark-mode dashboard using CustomTkinter
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

import customtkinter as ctk
import yaml
import threading
import time
from typing import Optional
from loguru import logger

from orchestrator import GreenInferenceOrchestrator, OrchestrationEvent


class GreenInferenceDashboard(ctk.CTk):
    """
    Modern dashboard for Green-Inference
    Shows real-time power metrics and AI performance
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        ui_config = config.get('UI', {})
        
        # Window setup
        self.title("Green-Inference: AI Power Orchestrator")
        self.geometry(ui_config.get('WINDOW_SIZE', '900x700'))
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme(ui_config.get('THEME', 'dark-blue'))
        
        # Initialize orchestrator
        self.orchestrator = GreenInferenceOrchestrator(config)
        self.orchestrator.register_event_callback(self.on_orchestration_event)
        
        # UI state
        self.is_monitoring = False
        self.event_log = []
        
        # Build UI
        self.build_ui()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info("Dashboard initialized")
    
    def build_ui(self):
        """Build the user interface"""
        # Main container
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        self.build_header(main_frame)
        
        # Power metrics section
        self.build_power_section(main_frame)
        
        # Model status section
        self.build_model_section(main_frame)
        
        # Energy savings section
        self.build_energy_section(main_frame)
        
        # AI inference section
        self.build_inference_section(main_frame)
        
        # Event log section
        self.build_event_log(main_frame)
        
        # Control buttons
        self.build_controls(main_frame)
    
    def build_header(self, parent):
        """Build header section"""
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🌱 Green-Inference",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(side="left", padx=10)
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="AI Power Orchestrator",
            font=ctk.CTkFont(size=16)
        )
        subtitle_label.pack(side="left", padx=10)
        
        # Status indicator
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="● Running",
            font=ctk.CTkFont(size=14),
            text_color="#00FF00"
        )
        self.status_label.pack(side="right", padx=10)
    
    def build_power_section(self, parent):
        """Build power metrics section"""
        power_frame = ctk.CTkFrame(parent)
        power_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        title = ctk.CTkLabel(
            power_frame,
            text="⚡ Power Metrics",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Battery
        self.battery_label = ctk.CTkLabel(power_frame, text="Battery: ---%")
        self.battery_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.battery_bar = ctk.CTkProgressBar(power_frame, width=300)
        self.battery_bar.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.battery_bar.set(0)
        
        # Power state
        self.power_state_label = ctk.CTkLabel(power_frame, text="Power State: ---")
        self.power_state_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        # CPU/GPU/Total power
        self.cpu_power_label = ctk.CTkLabel(power_frame, text="CPU Power: --- W")
        self.cpu_power_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        self.gpu_power_label = ctk.CTkLabel(power_frame, text="GPU Power: --- W")
        self.gpu_power_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        
        self.total_power_label = ctk.CTkLabel(power_frame, text="Total Power: --- W")
        self.total_power_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        
        power_frame.grid_columnconfigure(1, weight=1)
    
    def build_model_section(self, parent):
        """Build model status section"""
        model_frame = ctk.CTkFrame(parent)
        model_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        title = ctk.CTkLabel(
            model_frame,
            text="🤖 AI Model Status",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.model_name_label = ctk.CTkLabel(model_frame, text="Current Model: ---")
        self.model_name_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.model_swaps_label = ctk.CTkLabel(model_frame, text="Model Swaps: 0")
        self.model_swaps_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.inference_count_label = ctk.CTkLabel(model_frame, text="Inferences: 0")
        self.inference_count_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
    
    def build_energy_section(self, parent):
        """Build energy savings section"""
        energy_frame = ctk.CTkFrame(parent)
        energy_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        
        title = ctk.CTkLabel(
            energy_frame,
            text="💚 Energy Savings",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.energy_saved_label = ctk.CTkLabel(energy_frame, text="Energy Saved: 0.00 Wh")
        self.energy_saved_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.carbon_saved_label = ctk.CTkLabel(energy_frame, text="Carbon Offset: 0.0000 kg CO₂")
        self.carbon_saved_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
    
    def build_inference_section(self, parent):
        """Build AI inference testing section"""
        inference_frame = ctk.CTkFrame(parent)
        inference_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        
        title = ctk.CTkLabel(
            inference_frame,
            text="💬 Test AI Inference",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Input
        self.prompt_entry = ctk.CTkEntry(
            inference_frame,
            placeholder_text="Enter your prompt here...",
            width=400
        )
        self.prompt_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        # Generate button
        self.generate_btn = ctk.CTkButton(
            inference_frame,
            text="Generate",
            command=self.on_generate_click
        )
        self.generate_btn.grid(row=1, column=1, padx=10, pady=5)
        
        # Response
        self.response_text = ctk.CTkTextbox(inference_frame, height=100)
        self.response_text.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        inference_frame.grid_columnconfigure(0, weight=1)
    
    def build_event_log(self, parent):
        """Build event log section"""
        log_frame = ctk.CTkFrame(parent)
        log_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
        
        title = ctk.CTkLabel(
            log_frame,
            text="📋 Event Log",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.event_log_text = ctk.CTkTextbox(log_frame, height=100)
        self.event_log_text.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        log_frame.grid_columnconfigure(0, weight=1)
    
    def build_controls(self, parent):
        """Build control buttons"""
        control_frame = ctk.CTkFrame(parent, fg_color="transparent")
        control_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
        
        self.start_btn = ctk.CTkButton(
            control_frame,
            text="Start Monitoring",
            command=self.start_monitoring
        )
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="Stop Monitoring",
            command=self.stop_monitoring,
            fg_color="#FF4444"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        self.clear_log_btn = ctk.CTkButton(
            control_frame,
            text="Clear Log",
            command=self.clear_event_log
        )
        self.clear_log_btn.pack(side="left", padx=5)
    
    def start_monitoring(self):
        """Start the orchestrator and UI updates"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.orchestrator.start()
        
        # Start UI update loop
        self.update_ui()
        
        self.log_event("System started")
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop the orchestrator"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.orchestrator.stop()
        
        self.status_label.configure(text="● Stopped", text_color="#FF0000")
        self.log_event("System stopped")
        logger.info("Monitoring stopped")
    
    def update_ui(self):
        """Update UI with current metrics"""
        if not self.is_monitoring:
            return
        
        try:
            status = self.orchestrator.get_status()
            
            # Update battery
            battery = status['battery_percent']
            self.battery_label.configure(text=f"Battery: {battery:.1f}%")
            self.battery_bar.set(battery / 100)
            
            # Battery color
            if battery > 50:
                color = "#00FF00"
            elif battery > 20:
                color = "#FFAA00"
            else:
                color = "#FF0000"
            self.battery_bar.configure(progress_color=color)
            
            # Power state
            power_state = status['power_state']
            self.power_state_label.configure(text=f"Power State: {power_state}")
            
            # Power metrics
            self.cpu_power_label.configure(text=f"CPU Power: {status['cpu_power']:.1f} W")
            self.gpu_power_label.configure(text=f"GPU Power: {status['gpu_power']:.1f} W")
            self.total_power_label.configure(text=f"Total Power: {status['total_power']:.1f} W")
            
            # Model status
            model = status['current_model'] or "None"
            self.model_name_label.configure(text=f"Current Model: {model}")
            self.model_swaps_label.configure(text=f"Model Swaps: {status['model_swaps']}")
            self.inference_count_label.configure(text=f"Inferences: {status['inference_count']}")
            
            # Energy savings
            self.energy_saved_label.configure(text=f"Energy Saved: {status['energy_saved_wh']:.2f} Wh")
            self.carbon_saved_label.configure(text=f"Carbon Offset: {status['carbon_saved_kg']:.4f} kg CO₂")
            
        except Exception as e:
            logger.error(f"UI update error: {e}")
        
        # Schedule next update
        update_interval = self.config.get('UI', {}).get('UPDATE_INTERVAL', 5000)
        self.after(update_interval, self.update_ui)
    
    def on_generate_click(self):
        """Handle generate button click"""
        prompt = self.prompt_entry.get()
        
        if not prompt:
            self.response_text.delete("1.0", "end")
            self.response_text.insert("1.0", "Please enter a prompt")
            return
        
        # Disable button during generation
        self.generate_btn.configure(state="disabled", text="Generating...")
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", "Generating response...")
        
        # Generate in background thread
        def generate():
            try:
                response = self.orchestrator.generate(prompt)
                
                # Update UI in main thread
                self.after(0, lambda: self.update_response(response))
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.after(0, lambda: self.update_response(error_msg))
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
    
    def update_response(self, response: str):
        """Update response text (called from main thread)"""
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", response)
        self.generate_btn.configure(state="normal", text="Generate")
    
    def on_orchestration_event(self, event: OrchestrationEvent):
        """Handle orchestration events"""
        event_msg = f"[{event.event_type}] {event.details}"
        self.log_event(event_msg)
    
    def log_event(self, message: str):
        """Add event to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.event_log.append(log_entry)
        
        # Update log display
        self.event_log_text.insert("end", log_entry)
        self.event_log_text.see("end")
        
        # Keep only last 100 entries
        if len(self.event_log) > 100:
            self.event_log.pop(0)
            # Clear and repopulate
            self.event_log_text.delete("1.0", "end")
            for entry in self.event_log:
                self.event_log_text.insert("end", entry)
    
    def clear_event_log(self):
        """Clear the event log"""
        self.event_log.clear()
        self.event_log_text.delete("1.0", "end")
    
    def on_closing(self):
        """Handle window close"""
        logger.info("Closing dashboard")
        self.stop_monitoring()
        self.destroy()


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run dashboard
    app = GreenInferenceDashboard(config)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
