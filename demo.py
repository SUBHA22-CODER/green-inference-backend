#!/usr/bin/env python3
"""
Green-Inference: Standalone Demo
Works without network dependencies - demonstrates core functionality
"""

import time
import random
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

# Simple logger replacement
class Logger:
    @staticmethod
    def info(msg): print(f"[INFO] {msg}")
    @staticmethod
    def success(msg): print(f"[✓] {msg}")
    @staticmethod
    def warning(msg): print(f"[⚠] {msg}")
    @staticmethod
    def error(msg): print(f"[✗] {msg}")

logger = Logger()

# Power State Enum
class PowerState(Enum):
    HIGH_POWER = "HIGH_POWER"
    MEDIUM_POWER = "MEDIUM_POWER"
    LOW_POWER = "LOW_POWER"
    CRITICAL = "CRITICAL"
    AC_POWER = "AC_POWER"

# Model Profile
@dataclass
class ModelProfile:
    name: str
    quantization: str
    power_consumption: str
    max_tokens: int

# Simulated Power Metrics
@dataclass
class PowerMetrics:
    battery_percent: float
    is_plugged: bool
    cpu_percent: float
    cpu_power_est: float
    gpu_power_est: float
    total_power_est: float
    power_state: PowerState
    timestamp: float

class DemoOrchestrator:
    """Simplified orchestrator for demo purposes"""
    
    def __init__(self):
        self.profiles = {
            "HIGH": ModelProfile("High Quality (8-bit)", "Q8_0", "HIGH", 512),
            "MEDIUM": ModelProfile("Medium Quality (4-bit)", "Q4_K_M", "MEDIUM", 256),
            "LOW": ModelProfile("Low Power (Tiny)", "Q4_K_M", "LOW", 128)
        }
        self.current_profile = None
        self.battery_percent = 100.0
        self.is_running = False
        self.energy_saved_wh = 0.0
        self.inference_count = 0
        self.swap_count = 0
        
        logger.success("Green-Inference Demo Initialized")
    
    def get_current_metrics(self) -> PowerMetrics:
        """Simulate power metrics"""
        # Simulate battery drain
        if not self.is_running:
            self.battery_percent = max(10, self.battery_percent - random.uniform(0.1, 0.5))
        
        # Determine power state
        if self.battery_percent > 50:
            power_state = PowerState.HIGH_POWER
            cpu_power = 20.0
            gpu_power = 25.0
        elif self.battery_percent > 20:
            power_state = PowerState.MEDIUM_POWER
            cpu_power = 15.0
            gpu_power = 10.0
        else:
            power_state = PowerState.LOW_POWER
            cpu_power = 10.0
            gpu_power = 5.0
        
        cpu_usage = random.uniform(20, 40)
        
        return PowerMetrics(
            battery_percent=self.battery_percent,
            is_plugged=False,
            cpu_percent=cpu_usage,
            cpu_power_est=cpu_power,
            gpu_power_est=gpu_power,
            total_power_est=cpu_power + gpu_power,
            power_state=power_state,
            timestamp=time.time()
        )
    
    def select_profile(self, metrics: PowerMetrics) -> str:
        """Select appropriate profile based on battery"""
        if metrics.battery_percent > 50:
            return "HIGH"
        elif metrics.battery_percent > 20:
            return "MEDIUM"
        else:
            return "LOW"
    
    def swap_model(self, profile_name: str):
        """Simulate model swap"""
        if self.current_profile != profile_name:
            old_profile = self.current_profile
            self.current_profile = profile_name
            self.swap_count += 1
            
            profile = self.profiles[profile_name]
            logger.info(f"Model Swap: {old_profile or 'None'} → {profile_name}")
            logger.success(f"Loaded: {profile.name} ({profile.quantization})")
            
            # Simulate load time
            time.sleep(0.5)
    
    def generate(self, prompt: str) -> str:
        """Simulate AI inference"""
        if not self.current_profile:
            return "Error: No model loaded"
        
        profile = self.profiles[self.current_profile]
        
        # Simulate inference time
        inference_time = random.uniform(0.3, 1.2)
        time.sleep(inference_time)
        
        self.inference_count += 1
        
        # Simulate energy savings
        baseline_power = 45.0  # High quality model
        current_power = 10.0 if profile.power_consumption == "LOW" else \
                       25.0 if profile.power_consumption == "MEDIUM" else 45.0
        
        energy_saved = (baseline_power - current_power) * (inference_time / 3600)
        self.energy_saved_wh += max(0, energy_saved)
        
        # Generate mock response based on profile
        responses = {
            "HIGH": f"[High Quality Response]\n\nSustainable AI refers to the development and deployment of artificial intelligence systems that minimize environmental impact through efficient resource utilization, reduced energy consumption, and carbon-aware computing practices. This includes optimizing model architectures, using quantization techniques, and implementing power-aware scheduling.\n\n(Generated with {profile.name}, {inference_time:.2f}s)",
            
            "MEDIUM": f"[Medium Quality Response]\n\nSustainable AI focuses on reducing the environmental footprint of AI systems through efficient computing. Key approaches include model optimization, quantization, and power-aware deployment strategies.\n\n(Generated with {profile.name}, {inference_time:.2f}s)",
            
            "LOW": f"[Low Power Response]\n\nSustainable AI minimizes energy use through efficient models and power management.\n\n(Generated with {profile.name}, {inference_time:.2f}s)"
        }
        
        return responses.get(self.current_profile, "Response generated")
    
    def print_status(self, metrics: PowerMetrics):
        """Print current status"""
        print("\n" + "="*70)
        print("GREEN-INFERENCE STATUS")
        print("="*70)
        print(f"Battery: {metrics.battery_percent:.1f}%")
        print(f"Power State: {metrics.power_state.value}")
        print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
        print(f"CPU Power: {metrics.cpu_power_est:.1f}W")
        print(f"GPU Power: {metrics.gpu_power_est:.1f}W")
        print(f"Total Power: {metrics.total_power_est:.1f}W")
        print(f"\nCurrent Model: {self.profiles[self.current_profile].name if self.current_profile else 'None'}")
        print(f"Model Swaps: {self.swap_count}")
        print(f"Inferences: {self.inference_count}")
        print(f"\nEnergy Saved: {self.energy_saved_wh:.4f} Wh")
        print(f"Carbon Offset: {self.energy_saved_wh * 0.0005:.6f} kg CO₂")
        print("="*70)


def run_interactive_demo():
    """Run interactive demo"""
    orchestrator = DemoOrchestrator()
    
    print("\n" + "="*70)
    print("🌱 GREEN-INFERENCE: INTERACTIVE DEMO")
    print("="*70)
    print("\nCommands:")
    print("  status    - Show current status")
    print("  generate  - Generate AI response")
    print("  drain     - Simulate battery drain")
    print("  demo      - Run automated demo sequence")
    print("  quit      - Exit")
    print("="*70)
    
    # Initial load
    metrics = orchestrator.get_current_metrics()
    profile = orchestrator.select_profile(metrics)
    orchestrator.swap_model(profile)
    orchestrator.print_status(metrics)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "quit" or cmd == "exit":
                break
            
            elif cmd == "status":
                metrics = orchestrator.get_current_metrics()
                orchestrator.print_status(metrics)
            
            elif cmd == "generate":
                prompt = input("Enter prompt (or press Enter for default): ").strip()
                if not prompt:
                    prompt = "What is sustainable AI?"
                
                print(f"\n💬 Prompt: {prompt}")
                print("Generating response...\n")
                
                response = orchestrator.generate(prompt)
                print(response)
                
                # Check if model swap needed
                metrics = orchestrator.get_current_metrics()
                new_profile = orchestrator.select_profile(metrics)
                if new_profile != orchestrator.current_profile:
                    print(f"\n⚡ Battery at {metrics.battery_percent:.1f}% - swapping model...")
                    orchestrator.swap_model(new_profile)
            
            elif cmd == "drain":
                amount = input("Drain amount (%) [default: 10]: ").strip()
                try:
                    amount = float(amount) if amount else 10.0
                    orchestrator.battery_percent = max(5, orchestrator.battery_percent - amount)
                    print(f"Battery drained to {orchestrator.battery_percent:.1f}%")
                    
                    # Check for model swap
                    metrics = orchestrator.get_current_metrics()
                    new_profile = orchestrator.select_profile(metrics)
                    if new_profile != orchestrator.current_profile:
                        orchestrator.swap_model(new_profile)
                        orchestrator.print_status(metrics)
                except ValueError:
                    print("Invalid amount")
            
            elif cmd == "demo":
                run_automated_demo(orchestrator)
            
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted")
            break
    
    print("\n👋 Thanks for trying Green-Inference!")


def run_automated_demo(orchestrator: DemoOrchestrator):
    """Run automated demo sequence"""
    print("\n" + "="*70)
    print("🎬 RUNNING AUTOMATED DEMO")
    print("="*70)
    
    # Reset to high battery
    orchestrator.battery_percent = 100.0
    orchestrator.energy_saved_wh = 0.0
    orchestrator.inference_count = 0
    orchestrator.swap_count = 0
    
    prompts = [
        "What is sustainable AI?",
        "How can we reduce AI energy consumption?",
        "Explain power-aware computing"
    ]
    
    for i, battery_level in enumerate([100, 45, 15]):
        print(f"\n{'─'*70}")
        print(f"SCENARIO {i+1}: Battery at {battery_level}%")
        print('─'*70)
        
        orchestrator.battery_percent = battery_level
        metrics = orchestrator.get_current_metrics()
        
        # Select and swap model
        new_profile = orchestrator.select_profile(metrics)
        orchestrator.swap_model(new_profile)
        
        # Show status
        orchestrator.print_status(metrics)
        
        # Generate response
        print(f"\n💬 Prompt: {prompts[i]}")
        print("Generating...\n")
        response = orchestrator.generate(prompts[i])
        print(response)
        
        time.sleep(2)
    
    # Final summary
    print("\n" + "="*70)
    print("📊 DEMO COMPLETE - FINAL RESULTS")
    print("="*70)
    print(f"Total Inferences: {orchestrator.inference_count}")
    print(f"Model Swaps: {orchestrator.swap_count}")
    print(f"Energy Saved: {orchestrator.energy_saved_wh:.4f} Wh")
    print(f"Carbon Offset: {orchestrator.energy_saved_wh * 0.0005:.6f} kg CO₂")
    print("\n💡 Key Takeaway:")
    print("   Green-Inference automatically adjusted model quality based on battery,")
    print("   maintaining AI functionality while saving significant energy!")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        orchestrator = DemoOrchestrator()
        metrics = orchestrator.get_current_metrics()
        profile = orchestrator.select_profile(metrics)
        orchestrator.swap_model(profile)
        run_automated_demo(orchestrator)
    else:
        run_interactive_demo()
