"""
Green-Inference: Orchestrator
Main control loop that coordinates power monitoring and model swapping
"""

import time
import threading
from typing import Dict, Optional, Callable
from loguru import logger
from dataclasses import dataclass

from power_monitor import PowerMonitor, PowerState, PowerMetrics
from model_manager import ModelManager


@dataclass
class OrchestrationEvent:
    """Event triggered during orchestration"""
    event_type: str  # 'model_swap', 'power_change', 'inference'
    timestamp: float
    details: Dict


class GreenInferenceOrchestrator:
    """
    Main orchestrator for Green-Inference
    Coordinates power monitoring and intelligent model swapping
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.power_monitor = PowerMonitor(config)
        self.model_manager = ModelManager(config)
        
        # Orchestration settings
        self.check_interval = config.get('MONITORING', {}).get('CHECK_INTERVAL', 30)
        
        # State management
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_power_state: Optional[PowerState] = None
        self.event_callbacks: list = []
        
        # Energy tracking
        self.baseline_power_without_optimization = 0.0
        
        logger.info("GreenInferenceOrchestrator initialized")
    
    def register_event_callback(self, callback: Callable):
        """Register a callback for orchestration events"""
        self.event_callbacks.append(callback)
        logger.info(f"Registered event callback: {callback.__name__}")
    
    def _emit_event(self, event: OrchestrationEvent):
        """Emit an event to all registered callbacks"""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")
    
    def start(self):
        """Start the orchestration loop"""
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("Starting Green-Inference Orchestrator")
        self.is_running = True
        
        # Start monitoring in background thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.success("Orchestrator started successfully")
    
    def stop(self):
        """Stop the orchestration loop"""
        if not self.is_running:
            logger.warning("Orchestrator not running")
            return
        
        logger.info("Stopping Green-Inference Orchestrator")
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.success("Orchestrator stopped")
    
    def _monitoring_loop(self):
        """
        Background monitoring loop
        Continuously checks power state and swaps models as needed
        """
        logger.info("Monitoring loop started")
        
        # Initial model load
        metrics = self.power_monitor.get_metrics()
        self._handle_power_state_change(metrics)
        
        while self.is_running:
            try:
                # Get current power metrics
                metrics = self.power_monitor.get_metrics()
                
                # Check if power state changed
                if self.last_power_state != metrics.power_state:
                    logger.info(
                        f"Power state changed: {self.last_power_state} → {metrics.power_state}"
                    )
                    
                    self._handle_power_state_change(metrics)
                    self.last_power_state = metrics.power_state
                    
                    # Emit event
                    event = OrchestrationEvent(
                        event_type='power_change',
                        timestamp=time.time(),
                        details={
                            'old_state': self.last_power_state.value if self.last_power_state else None,
                            'new_state': metrics.power_state.value,
                            'battery_percent': metrics.battery_percent
                        }
                    )
                    self._emit_event(event)
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Short sleep before retry
        
        logger.info("Monitoring loop stopped")
    
    def _handle_power_state_change(self, metrics: PowerMetrics):
        """
        Handle power state changes by swapping models if needed
        
        Args:
            metrics: Current power metrics
        """
        logger.info(f"Handling power state: {metrics.power_state.value}")
        logger.info(f"Battery: {metrics.battery_percent:.1f}%, Plugged: {metrics.is_plugged}")
        
        # Determine if model swap is needed
        swapped = self.model_manager.swap_if_needed(
            metrics.battery_percent,
            metrics.is_plugged
        )
        
        if swapped:
            logger.success(f"Model swapped to: {self.model_manager.current_profile.name}")
            
            # Track energy savings
            current_profile = self.model_manager.current_profile
            if current_profile.power_consumption == "LOW":
                self.baseline_power_without_optimization = 45  # High power model estimate
            elif current_profile.power_consumption == "MEDIUM":
                self.baseline_power_without_optimization = 35
            else:
                self.baseline_power_without_optimization = metrics.total_power_est
            
            # Emit event
            event = OrchestrationEvent(
                event_type='model_swap',
                timestamp=time.time(),
                details={
                    'new_model': current_profile.name,
                    'quantization': current_profile.quantization,
                    'power_level': current_profile.power_consumption,
                    'battery_percent': metrics.battery_percent
                }
            )
            self._emit_event(event)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate AI response with power tracking
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        if not self.is_running:
            logger.warning("Orchestrator not running, starting it now")
            self.start()
            time.sleep(2)  # Give it time to load model
        
        # Get current metrics before inference
        metrics_before = self.power_monitor.get_metrics()
        start_time = time.time()
        
        # Generate response
        response = self.model_manager.generate(prompt, max_tokens)
        
        # Get metrics after inference
        metrics_after = self.power_monitor.get_metrics()
        inference_time = time.time() - start_time
        
        # Track energy consumption
        avg_power = (metrics_before.total_power_est + metrics_after.total_power_est) / 2
        self.power_monitor.track_energy_savings(
            self.baseline_power_without_optimization,
            avg_power,
            inference_time
        )
        
        # Emit event
        event = OrchestrationEvent(
            event_type='inference',
            timestamp=time.time(),
            details={
                'prompt_length': len(prompt),
                'response_length': len(response),
                'inference_time': inference_time,
                'power_during_inference': avg_power,
                'model': self.model_manager.current_profile.name if self.model_manager.current_profile else None
            }
        )
        self._emit_event(event)
        
        return response
    
    def get_status(self) -> Dict:
        """Get current orchestrator status"""
        current_metrics = self.power_monitor.get_metrics()
        model_stats = self.model_manager.get_stats()
        energy_stats = self.power_monitor.get_energy_stats()
        
        return {
            'is_running': self.is_running,
            'current_model': model_stats['current_model'],
            'power_state': current_metrics.power_state.value,
            'battery_percent': current_metrics.battery_percent,
            'is_plugged': current_metrics.is_plugged,
            'cpu_power': current_metrics.cpu_power_est,
            'gpu_power': current_metrics.gpu_power_est,
            'total_power': current_metrics.total_power_est,
            'energy_saved_wh': energy_stats['energy_saved_wh'],
            'carbon_saved_kg': energy_stats['carbon_saved_kg'],
            'inference_count': model_stats['inference_count'],
            'model_swaps': model_stats['swap_count']
        }
    
    def get_detailed_status(self) -> str:
        """Get formatted detailed status"""
        status = self.get_status()
        
        lines = [
            "=" * 60,
            "GREEN-INFERENCE ORCHESTRATOR STATUS",
            "=" * 60,
            f"Running: {status['is_running']}",
            f"Current Model: {status['current_model']}",
            f"Power State: {status['power_state']}",
            "",
            "Power Metrics:",
            f"  Battery: {status['battery_percent']:.1f}%",
            f"  AC Power: {'Yes' if status['is_plugged'] else 'No'}",
            f"  CPU Power: {status['cpu_power']:.1f}W",
            f"  GPU Power: {status['gpu_power']:.1f}W",
            f"  Total Power: {status['total_power']:.1f}W",
            "",
            "Performance:",
            f"  Inferences: {status['inference_count']}",
            f"  Model Swaps: {status['model_swaps']}",
            "",
            "Energy Savings:",
            f"  Energy Saved: {status['energy_saved_wh']:.2f} Wh",
            f"  Carbon Offset: {status['carbon_saved_kg']:.4f} kg CO2",
            "=" * 60
        ]
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the orchestrator
    import yaml
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = GreenInferenceOrchestrator(config)
    
    # Event callback for logging
    def log_event(event: OrchestrationEvent):
        logger.info(f"Event: {event.event_type} - {event.details}")
    
    orchestrator.register_event_callback(log_event)
    
    # Start orchestrator
    orchestrator.start()
    
    print("\n" + orchestrator.get_detailed_status())
    
    # Test inference
    print("\n" + "=" * 60)
    print("Testing AI inference...")
    print("=" * 60)
    
    response = orchestrator.generate("What is sustainable AI?")
    print(f"\nResponse:\n{response}")
    
    # Monitor for 30 seconds
    print("\n" + "=" * 60)
    print("Monitoring for 30 seconds (Ctrl+C to stop)...")
    print("=" * 60)
    
    try:
        for i in range(6):
            time.sleep(5)
            print(f"\n[{i+1}/6] " + orchestrator.get_detailed_status())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Stop orchestrator
    orchestrator.stop()
    
    print("\n" + orchestrator.get_detailed_status())
