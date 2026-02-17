"""
Green-Inference: Power Monitor Module
Monitors battery level, CPU/GPU/NPU power consumption in real-time
"""

import psutil
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil not available - GPU monitoring disabled")


class PowerState(Enum):
    """System power states based on battery level"""
    HIGH_POWER = "HIGH_POWER"      # > 50% battery
    MEDIUM_POWER = "MEDIUM_POWER"  # 20-50% battery
    LOW_POWER = "LOW_POWER"        # 10-20% battery
    CRITICAL = "CRITICAL"          # < 10% battery
    AC_POWER = "AC_POWER"          # Plugged in


@dataclass
class PowerMetrics:
    """Current power consumption metrics"""
    battery_percent: float
    battery_time_left: Optional[int]  # seconds
    is_plugged: bool
    cpu_percent: float
    cpu_freq: float  # MHz
    cpu_power_est: float  # Watts (estimated)
    gpu_power_est: float  # Watts (estimated)
    npu_power_est: float  # Watts (estimated - AMD Ryzen AI)
    total_power_est: float  # Watts
    power_state: PowerState
    timestamp: float


class PowerMonitor:
    """
    Real-time power monitoring for AI workload optimization
    Tracks battery, CPU, GPU, and NPU power consumption
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.thresholds = config.get('POWER_THRESHOLDS', {})
        self.hardware = config.get('HARDWARE', {})
        self.energy_config = config.get('ENERGY', {})
        
        # Power baselines (watts)
        self.baseline_cpu = self.energy_config.get('BASELINE_POWER_CPU', 15)
        self.baseline_gpu = self.energy_config.get('BASELINE_POWER_GPU', 30)
        self.baseline_npu = self.energy_config.get('BASELINE_POWER_NPU', 5)
        
        # Monitoring state
        self.last_metrics: Optional[PowerMetrics] = None
        self.energy_saved_wh = 0.0  # Watt-hours saved
        self.monitoring_start = time.time()
        
        logger.info("PowerMonitor initialized")
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system information on startup"""
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        ram = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"CPU: {cpu_count} physical cores, {cpu_count_logical} logical cores")
        logger.info(f"RAM: {ram:.1f} GB")
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    logger.info(f"GPU: {gpu.name} ({gpu.memoryTotal}MB)")
            except:
                logger.warning("GPU detection failed")
    
    def get_battery_info(self) -> Tuple[float, Optional[int], bool]:
        """
        Get current battery status
        Returns: (percent, seconds_left, is_plugged)
        """
        battery = psutil.sensors_battery()
        
        if battery is None:
            # Desktop or VM - assume AC power
            return 100.0, None, True
        
        return (
            battery.percent,
            battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None,
            battery.power_plugged
        )
    
    def estimate_cpu_power(self) -> float:
        """
        Estimate CPU power consumption based on utilization
        Returns: Estimated watts
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Simple linear model: Power = Baseline * (1 + utilization%)
        # More accurate models would use CPU package power from sensors
        power_multiplier = 1 + (cpu_percent / 100)
        estimated_power = self.baseline_cpu * power_multiplier
        
        # Frequency scaling adjustment
        if cpu_freq and cpu_freq.max > 0:
            freq_ratio = cpu_freq.current / cpu_freq.max
            estimated_power *= (0.5 + 0.5 * freq_ratio)  # Power scales ~quadratically with freq
        
        return estimated_power
    
    def estimate_gpu_power(self) -> float:
        """
        Estimate GPU power consumption
        Returns: Estimated watts
        """
        if not GPU_AVAILABLE:
            return 0.0
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0.0
            
            # Use first GPU
            gpu = gpus[0]
            gpu_load = gpu.load * 100  # 0-100%
            
            # Simple model: baseline + load-based consumption
            estimated_power = self.baseline_gpu * (gpu_load / 100)
            
            return estimated_power
        except Exception as e:
            logger.warning(f"GPU power estimation failed: {e}")
            return 0.0
    
    def estimate_npu_power(self) -> float:
        """
        Estimate NPU power consumption (AMD Ryzen AI)
        Returns: Estimated watts
        
        Note: NPU monitoring is AMD-specific and requires hardware access
        This is a placeholder for future AMD NPU integration
        """
        # Placeholder: In production, this would query AMD Ryzen AI NPU status
        # For now, return 0 (NPU not actively used in this demo)
        return 0.0
    
    def determine_power_state(self, battery_percent: float, is_plugged: bool) -> PowerState:
        """Determine current power state based on battery level"""
        if is_plugged:
            return PowerState.AC_POWER
        
        high_threshold = self.thresholds.get('HIGH_POWER', 50)
        medium_threshold = self.thresholds.get('MEDIUM_POWER', 20)
        critical_threshold = self.thresholds.get('CRITICAL_POWER', 10)
        
        if battery_percent > high_threshold:
            return PowerState.HIGH_POWER
        elif battery_percent > medium_threshold:
            return PowerState.MEDIUM_POWER
        elif battery_percent > critical_threshold:
            return PowerState.LOW_POWER
        else:
            return PowerState.CRITICAL
    
    def get_metrics(self) -> PowerMetrics:
        """
        Get current power metrics
        This is the main method called by the orchestrator
        """
        # Battery info
        battery_percent, battery_time, is_plugged = self.get_battery_info()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
        
        # Power estimates
        cpu_power = self.estimate_cpu_power()
        gpu_power = self.estimate_gpu_power()
        npu_power = self.estimate_npu_power()
        total_power = cpu_power + gpu_power + npu_power
        
        # Power state
        power_state = self.determine_power_state(battery_percent, is_plugged)
        
        metrics = PowerMetrics(
            battery_percent=battery_percent,
            battery_time_left=battery_time,
            is_plugged=is_plugged,
            cpu_percent=cpu_percent,
            cpu_freq=cpu_freq_mhz,
            cpu_power_est=cpu_power,
            gpu_power_est=gpu_power,
            npu_power_est=npu_power,
            total_power_est=total_power,
            power_state=power_state,
            timestamp=time.time()
        )
        
        self.last_metrics = metrics
        return metrics
    
    def track_energy_savings(self, baseline_power: float, actual_power: float, duration_sec: float):
        """
        Track energy savings from model swapping
        
        Args:
            baseline_power: Power consumption without optimization (watts)
            actual_power: Actual power consumption (watts)
            duration_sec: Duration of measurement (seconds)
        """
        power_saved = baseline_power - actual_power
        if power_saved > 0:
            energy_saved = (power_saved * duration_sec) / 3600  # Convert to Wh
            self.energy_saved_wh += energy_saved
            
            logger.info(f"Energy saved: {energy_saved:.2f} Wh (Total: {self.energy_saved_wh:.2f} Wh)")
    
    def get_energy_stats(self) -> Dict:
        """Get energy savings statistics"""
        runtime_hours = (time.time() - self.monitoring_start) / 3600
        carbon_intensity = self.energy_config.get('CARBON_INTENSITY', 0.5)
        carbon_saved_kg = (self.energy_saved_wh / 1000) * carbon_intensity
        
        return {
            'energy_saved_wh': self.energy_saved_wh,
            'runtime_hours': runtime_hours,
            'carbon_saved_kg': carbon_saved_kg,
            'avg_power_saved_w': (self.energy_saved_wh / runtime_hours) if runtime_hours > 0 else 0
        }
    
    def format_metrics(self, metrics: PowerMetrics) -> str:
        """Format metrics for display"""
        lines = [
            f"Battery: {metrics.battery_percent:.1f}%",
            f"Power State: {metrics.power_state.value}",
            f"AC Plugged: {'Yes' if metrics.is_plugged else 'No'}",
            f"CPU Usage: {metrics.cpu_percent:.1f}%",
            f"CPU Freq: {metrics.cpu_freq:.0f} MHz",
            f"CPU Power: {metrics.cpu_power_est:.1f}W",
            f"GPU Power: {metrics.gpu_power_est:.1f}W",
            f"NPU Power: {metrics.npu_power_est:.1f}W",
            f"Total Power: {metrics.total_power_est:.1f}W"
        ]
        
        if metrics.battery_time_left:
            hours = metrics.battery_time_left // 3600
            minutes = (metrics.battery_time_left % 3600) // 60
            lines.insert(2, f"Time Left: {hours}h {minutes}m")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the power monitor
    import yaml
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    monitor = PowerMonitor(config)
    
    print("Starting power monitoring (Ctrl+C to stop)...\n")
    try:
        while True:
            metrics = monitor.get_metrics()
            print("\033[H\033[J")  # Clear screen
            print("=" * 50)
            print("GREEN-INFERENCE POWER MONITOR")
            print("=" * 50)
            print(monitor.format_metrics(metrics))
            print("=" * 50)
            
            energy_stats = monitor.get_energy_stats()
            print(f"\nEnergy Saved: {energy_stats['energy_saved_wh']:.2f} Wh")
            print(f"Carbon Offset: {energy_stats['carbon_saved_kg']:.4f} kg CO2")
            
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
