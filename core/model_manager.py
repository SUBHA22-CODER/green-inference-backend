"""
Green-Inference: Model Manager
Handles dynamic loading, swapping, and inference of AI models
Optimized for power-aware operation
"""

import os
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - using mock models")


@dataclass
class ModelConfig:
    """Configuration for a specific model profile"""
    name: str
    model_id: str
    model_file: str
    quantization: str
    max_tokens: int
    temperature: float
    power_consumption: str
    battery_threshold: int


class ModelManager:
    """
    Manages AI model lifecycle with power-aware swapping
    Supports hot-swapping between different quantization levels
    """
    
    def __init__(self, config: Dict, cache_dir: Optional[str] = None):
        self.config = config
        self.model_profiles = self._parse_model_profiles()
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/green-inference")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Current model state
        self.current_model: Optional[object] = None
        self.current_tokenizer: Optional[object] = None
        self.current_profile: Optional[ModelConfig] = None
        self.model_load_time: float = 0.0
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.swap_count = 0
        
        logger.info(f"ModelManager initialized with {len(self.model_profiles)} profiles")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _parse_model_profiles(self) -> Dict[str, ModelConfig]:
        """Parse model profiles from config"""
        profiles = {}
        
        for profile_name, profile_data in self.config.get('MODEL_PROFILES', {}).items():
            profiles[profile_name] = ModelConfig(
                name=profile_data['name'],
                model_id=profile_data['model_id'],
                model_file=profile_data['model_file'],
                quantization=profile_data['quantization'],
                max_tokens=profile_data['max_tokens'],
                temperature=profile_data['temperature'],
                power_consumption=profile_data['power_consumption'],
                battery_threshold=profile_data['battery_threshold']
            )
        
        return profiles
    
    def get_profile_for_battery(self, battery_percent: float, is_plugged: bool) -> str:
        """
        Determine which model profile to use based on battery level
        
        Args:
            battery_percent: Current battery percentage
            is_plugged: Whether AC power is connected
            
        Returns:
            Profile name (HIGH_QUALITY, MEDIUM_QUALITY, or LOW_QUALITY)
        """
        if is_plugged:
            return "HIGH_QUALITY"
        
        if battery_percent > 50:
            return "HIGH_QUALITY"
        elif battery_percent > 20:
            return "MEDIUM_QUALITY"
        else:
            return "LOW_QUALITY"
    
    def load_model(self, profile_name: str) -> bool:
        """
        Load a model based on profile name
        
        Args:
            profile_name: Name of the profile to load
            
        Returns:
            True if successful, False otherwise
        """
        if profile_name not in self.model_profiles:
            logger.error(f"Unknown profile: {profile_name}")
            return False
        
        profile = self.model_profiles[profile_name]
        
        # Check if already loaded
        if self.current_profile and self.current_profile.name == profile.name:
            logger.info(f"Model {profile.name} already loaded")
            return True
        
        # Unload current model
        if self.current_model is not None:
            logger.info(f"Unloading current model: {self.current_profile.name}")
            self._unload_model()
        
        # Load new model
        logger.info(f"Loading model: {profile.name}")
        logger.info(f"  Model ID: {profile.model_id}")
        logger.info(f"  Quantization: {profile.quantization}")
        logger.info(f"  Power Level: {profile.power_consumption}")
        
        start_time = time.time()
        
        try:
            if TRANSFORMERS_AVAILABLE:
                # Use real models
                self._load_real_model(profile)
            else:
                # Use mock for development/testing
                self._load_mock_model(profile)
            
            self.model_load_time = time.time() - start_time
            self.current_profile = profile
            self.swap_count += 1
            
            logger.success(f"Model loaded in {self.model_load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_real_model(self, profile: ModelConfig):
        """Load actual transformer model"""
        # For GGUF models, we'd use llama-cpp-python
        # For standard HF models, use transformers
        
        # Simplified version using TinyLlama for demo
        if "TinyLlama" in profile.model_id:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            logger.info(f"Loading tokenizer from {model_name}")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loading model from {model_name}")
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if profile.quantization.startswith("Q8") else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # For Llama-2 GGUF models, use llama-cpp-python (not implemented here)
            # Fallback to mock
            logger.warning(f"GGUF loading not implemented, using mock for {profile.model_id}")
            self._load_mock_model(profile)
    
    def _load_mock_model(self, profile: ModelConfig):
        """Load mock model for testing"""
        self.current_model = {
            'type': 'mock',
            'profile': profile.name,
            'quantization': profile.quantization
        }
        self.current_tokenizer = {
            'type': 'mock'
        }
        logger.info("Mock model loaded (for testing)")
    
    def _unload_model(self):
        """Unload current model to free memory"""
        if self.current_model is not None:
            if TRANSFORMERS_AVAILABLE and hasattr(self.current_model, 'cpu'):
                self.current_model.cpu()
                del self.current_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.current_model = None
            self.current_tokenizer = None
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Generate text using the current model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate (overrides profile default)
            
        Returns:
            Generated text
        """
        if self.current_model is None:
            logger.error("No model loaded")
            return "Error: No model loaded"
        
        if self.current_profile is None:
            logger.error("No profile loaded")
            return "Error: No profile loaded"
        
        max_tokens = max_new_tokens or self.current_profile.max_tokens
        
        start_time = time.time()
        
        try:
            if isinstance(self.current_model, dict) and self.current_model.get('type') == 'mock':
                # Mock generation
                response = self._generate_mock(prompt, max_tokens)
            else:
                # Real generation
                response = self._generate_real(prompt, max_tokens)
            
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            logger.info(f"Generated {len(response)} chars in {inference_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"
    
    def _generate_real(self, prompt: str, max_tokens: int) -> str:
        """Generate using real model"""
        inputs = self.current_tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.current_profile.temperature,
                do_sample=True,
                pad_token_id=self.current_tokenizer.eos_token_id
            )
        
        response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def _generate_mock(self, prompt: str, max_tokens: int) -> str:
        """Generate mock response for testing"""
        time.sleep(0.5)  # Simulate processing
        
        profile_name = self.current_profile.name
        quant = self.current_profile.quantization
        
        return (
            f"[Mock Response from {profile_name} ({quant})]\n\n"
            f"This is a simulated AI response for testing purposes. "
            f"The actual model would generate meaningful text here. "
            f"Prompt: '{prompt[:50]}...'\n\n"
            f"Max tokens: {max_tokens}, Temperature: {self.current_profile.temperature}"
        )
    
    def swap_if_needed(self, battery_percent: float, is_plugged: bool) -> bool:
        """
        Check if model should be swapped based on battery level
        
        Args:
            battery_percent: Current battery percentage
            is_plugged: Whether AC power is connected
            
        Returns:
            True if model was swapped, False otherwise
        """
        target_profile = self.get_profile_for_battery(battery_percent, is_plugged)
        
        if self.current_profile is None:
            # No model loaded, load the target
            logger.info(f"No model loaded, loading {target_profile}")
            return self.load_model(target_profile)
        
        if self.current_profile and target_profile != self._get_profile_key(self.current_profile):
            # Need to swap
            logger.info(f"Swapping model: {self.current_profile.name} → {target_profile}")
            return self.load_model(target_profile)
        
        return False
    
    def _get_profile_key(self, profile: ModelConfig) -> str:
        """Get profile key from ModelConfig"""
        for key, val in self.model_profiles.items():
            if val.name == profile.name:
                return key
        return ""
    
    def get_stats(self) -> Dict:
        """Get model manager statistics"""
        avg_inference_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else 0
        )
        
        return {
            'current_model': self.current_profile.name if self.current_profile else None,
            'model_load_time': self.model_load_time,
            'inference_count': self.inference_count,
            'avg_inference_time': avg_inference_time,
            'swap_count': self.swap_count,
            'total_inference_time': self.total_inference_time
        }
    
    def list_profiles(self) -> List[str]:
        """List available model profiles"""
        return [
            f"{key}: {profile.name} ({profile.quantization})"
            for key, profile in self.model_profiles.items()
        ]


if __name__ == "__main__":
    # Test the model manager
    import yaml
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    manager = ModelManager(config)
    
    print("Available profiles:")
    for profile in manager.list_profiles():
        print(f"  - {profile}")
    
    print("\nLoading LOW_QUALITY model for testing...")
    manager.load_model("LOW_QUALITY")
    
    print("\nGenerating test response...")
    response = manager.generate("Hello, how are you?")
    print(f"\nResponse:\n{response}")
    
    print("\nStats:")
    stats = manager.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
