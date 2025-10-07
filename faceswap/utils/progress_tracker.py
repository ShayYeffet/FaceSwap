"""
Progress tracking utilities for FaceSwap application.
"""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any, List
from tqdm import tqdm
from datetime import datetime, timedelta
import sys

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks and displays progress for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps (int): Total number of steps to complete
            description (str): Description of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = None
        self.progress_bar = None
        self._last_update_time = 0
        self._update_interval = 1.0  # Update every second
    
    def start(self):
        """Start the progress tracking."""
        self.start_time = time.time()
        self.progress_bar = tqdm(
            total=self.total_steps,
            desc=self.description,
            unit="step",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        logger.info(f"Started {self.description} - {self.total_steps} steps")
    
    def update(self, step_increment: int = 1, message: Optional[str] = None):
        """
        Update progress by the specified number of steps.
        
        Args:
            step_increment (int): Number of steps to increment
            message (str, optional): Additional message to display
        """
        if self.progress_bar is None:
            self.start()
        
        self.current_step += step_increment
        self.progress_bar.update(step_increment)
        
        # Update description with message if provided
        if message:
            current_time = time.time()
            if current_time - self._last_update_time >= self._update_interval:
                self.progress_bar.set_postfix_str(message)
                self._last_update_time = current_time
    
    def set_step(self, step: int, message: Optional[str] = None):
        """
        Set the current step directly.
        
        Args:
            step (int): Current step number
            message (str, optional): Additional message to display
        """
        if self.progress_bar is None:
            self.start()
        
        step_increment = step - self.current_step
        if step_increment > 0:
            self.update(step_increment, message)
    
    def finish(self, success_message: Optional[str] = None):
        """
        Finish the progress tracking.
        
        Args:
            success_message (str, optional): Success message to display
        """
        if self.progress_bar is not None:
            self.progress_bar.close()
        
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed_time)
            
            if success_message:
                logger.info(f"{success_message} (completed in {elapsed_str})")
            else:
                logger.info(f"{self.description} completed in {elapsed_str}")
    
    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since start.
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_estimated_remaining_time(self) -> float:
        """
        Get estimated remaining time based on current progress.
        
        Returns:
            float: Estimated remaining time in seconds
        """
        if self.start_time is None or self.current_step == 0:
            return 0.0
        
        elapsed_time = self.get_elapsed_time()
        time_per_step = elapsed_time / self.current_step
        remaining_steps = self.total_steps - self.current_step
        
        return time_per_step * remaining_steps
    
    def get_progress_percentage(self) -> float:
        """
        Get current progress as percentage.
        
        Returns:
            float: Progress percentage (0-100)
        """
        if self.total_steps == 0:
            return 100.0
        return (self.current_step / self.total_steps) * 100
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"


class TrainingProgressTracker(ProgressTracker):
    """Specialized progress tracker for training operations."""
    
    def __init__(self, total_epochs: int, steps_per_epoch: int):
        """
        Initialize training progress tracker.
        
        Args:
            total_epochs (int): Total number of training epochs
            steps_per_epoch (int): Number of steps per epoch
        """
        super().__init__(total_epochs * steps_per_epoch, "Training")
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.current_epoch_step = 0
        self.best_loss = float('inf')
        self.losses = []
    
    def update_epoch(self, epoch: int, epoch_step: int, loss: float):
        """
        Update progress for training epoch.
        
        Args:
            epoch (int): Current epoch number
            epoch_step (int): Current step within epoch
            loss (float): Current loss value
        """
        self.current_epoch = epoch
        self.current_epoch_step = epoch_step
        self.losses.append(loss)
        
        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Calculate overall step
        overall_step = epoch * self.steps_per_epoch + epoch_step
        
        # Create progress message
        avg_loss = sum(self.losses[-10:]) / min(len(self.losses), 10)  # Last 10 losses
        message = f"Epoch {epoch+1}/{self.total_epochs}, Loss: {loss:.4f}, Avg: {avg_loss:.4f}, Best: {self.best_loss:.4f}"
        
        self.set_step(overall_step, message)
    
    def epoch_complete(self, epoch: int, epoch_loss: float):
        """
        Mark epoch as complete.
        
        Args:
            epoch (int): Completed epoch number
            epoch_loss (float): Average loss for the epoch
        """
        logger.info(f"Epoch {epoch+1}/{self.total_epochs} completed - Average loss: {epoch_loss:.4f}")


class VideoProcessingTracker(ProgressTracker):
    """Specialized progress tracker for video processing operations."""
    
    def __init__(self, total_frames: int):
        """
        Initialize video processing tracker.
        
        Args:
            total_frames (int): Total number of frames to process
        """
        super().__init__(total_frames, "Processing Video")
        self.total_frames = total_frames
        self.processed_frames = 0
        self.faces_detected = 0
        self.faces_swapped = 0
    
    def update_frame(self, faces_detected: int = 0, faces_swapped: int = 0):
        """
        Update progress for processed frame.
        
        Args:
            faces_detected (int): Number of faces detected in frame
            faces_swapped (int): Number of faces swapped in frame
        """
        self.processed_frames += 1
        self.faces_detected += faces_detected
        self.faces_swapped += faces_swapped
        
        # Calculate processing rate
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0
            
            message = f"Frame {self.processed_frames}/{self.total_frames}, " \
                     f"Faces: {self.faces_detected}, Swapped: {self.faces_swapped}, " \
                     f"FPS: {fps:.1f}"
        else:
            message = f"Frame {self.processed_frames}/{self.total_frames}"
        
        self.update(1, message)
    
    def get_summary(self) -> dict:
        """
        Get processing summary statistics.
        
        Returns:
            dict: Summary statistics
        """
        elapsed_time = self.get_elapsed_time()
        fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'faces_detected': self.faces_detected,
            'faces_swapped': self.faces_swapped,
            'processing_time': elapsed_time,
            'average_fps': fps,
            'detection_rate': self.faces_detected / self.processed_frames if self.processed_frames > 0 else 0
        }


class MultiStageProgressTracker:
    """
    Advanced progress tracker for multi-stage operations with detailed feedback.
    """
    
    def __init__(self, stages: List[Dict[str, Any]]):
        """
        Initialize multi-stage progress tracker.
        
        Args:
            stages: List of stage dictionaries with 'name', 'steps', and optional 'weight'
        """
        self.stages = stages
        self.current_stage_index = 0
        self.current_stage_progress = 0
        self.overall_progress = 0
        self.start_time = None
        self.stage_start_time = None
        self.progress_bar = None
        self.stage_times = []
        self.status_messages = []
        
        # Calculate total steps and weights
        self.total_steps = sum(stage.get('steps', 0) for stage in stages)
        total_weight = sum(stage.get('weight', 1) for stage in stages)
        
        # Normalize weights
        for stage in self.stages:
            stage['normalized_weight'] = stage.get('weight', 1) / total_weight
        
        logger.info(f"Multi-stage tracker initialized with {len(stages)} stages, {self.total_steps} total steps")
    
    def start(self):
        """Start the multi-stage progress tracking."""
        self.start_time = time.time()
        self.stage_start_time = self.start_time
        
        # Create overall progress bar
        self.progress_bar = tqdm(
            total=100,  # Use percentage for overall progress
            desc="Overall Progress",
            unit="%",
            ncols=100,
            bar_format='{l_bar}{bar}| {n:.1f}% [{elapsed}<{remaining}] {postfix}'
        )
        
        # Start first stage
        if self.stages:
            self._start_current_stage()
        
        logger.info("Multi-stage progress tracking started")
    
    def _start_current_stage(self):
        """Start the current stage."""
        if self.current_stage_index < len(self.stages):
            stage = self.stages[self.current_stage_index]
            self.stage_start_time = time.time()
            self.current_stage_progress = 0
            
            stage_msg = f"Stage {self.current_stage_index + 1}/{len(self.stages)}: {stage['name']}"
            self.progress_bar.set_postfix_str(stage_msg)
            
            logger.info(f"Started {stage_msg}")
    
    def update_stage_progress(self, steps: int = 1, message: Optional[str] = None):
        """
        Update progress within the current stage.
        
        Args:
            steps: Number of steps completed in current stage
            message: Optional status message
        """
        if self.current_stage_index >= len(self.stages):
            return
        
        stage = self.stages[self.current_stage_index]
        self.current_stage_progress += steps
        
        # Calculate stage completion percentage
        stage_completion = min(self.current_stage_progress / stage['steps'], 1.0) if stage['steps'] > 0 else 1.0
        
        # Calculate overall progress
        completed_stages_weight = sum(
            self.stages[i]['normalized_weight'] 
            for i in range(self.current_stage_index)
        )
        current_stage_contribution = stage['normalized_weight'] * stage_completion
        self.overall_progress = (completed_stages_weight + current_stage_contribution) * 100
        
        # Update progress bar
        progress_increment = self.overall_progress - self.progress_bar.n
        if progress_increment > 0:
            self.progress_bar.update(progress_increment)
        
        # Update status message
        if message:
            stage_msg = f"Stage {self.current_stage_index + 1}/{len(self.stages)}: {stage['name']} - {message}"
            self.progress_bar.set_postfix_str(stage_msg)
            self.status_messages.append({
                'timestamp': time.time(),
                'stage': self.current_stage_index,
                'message': message
            })
    
    def complete_stage(self, success_message: Optional[str] = None):
        """
        Mark the current stage as complete and move to next stage.
        
        Args:
            success_message: Optional success message for the completed stage
        """
        if self.current_stage_index >= len(self.stages):
            return
        
        stage = self.stages[self.current_stage_index]
        stage_duration = time.time() - self.stage_start_time
        self.stage_times.append({
            'stage_index': self.current_stage_index,
            'stage_name': stage['name'],
            'duration': stage_duration,
            'success_message': success_message
        })
        
        if success_message:
            logger.info(f"Stage {self.current_stage_index + 1} completed: {success_message} ({stage_duration:.1f}s)")
        else:
            logger.info(f"Stage {self.current_stage_index + 1} completed: {stage['name']} ({stage_duration:.1f}s)")
        
        # Move to next stage
        self.current_stage_index += 1
        if self.current_stage_index < len(self.stages):
            self._start_current_stage()
        else:
            # All stages complete
            self.overall_progress = 100
            self.progress_bar.update(100 - self.progress_bar.n)
            self.progress_bar.set_postfix_str("All stages completed")
    
    def get_estimated_remaining_time(self) -> float:
        """
        Get estimated remaining time based on completed stages.
        
        Returns:
            float: Estimated remaining time in seconds
        """
        if not self.start_time or self.overall_progress <= 0:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        time_per_percent = elapsed_time / self.overall_progress
        remaining_percent = 100 - self.overall_progress
        
        return time_per_percent * remaining_percent
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """
        Get summary of all stages and their completion status.
        
        Returns:
            Dict with stage summary information
        """
        return {
            'total_stages': len(self.stages),
            'completed_stages': len(self.stage_times),
            'current_stage': self.current_stage_index + 1 if self.current_stage_index < len(self.stages) else len(self.stages),
            'overall_progress_percent': self.overall_progress,
            'total_elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'estimated_remaining_time': self.get_estimated_remaining_time(),
            'stage_times': self.stage_times,
            'recent_messages': self.status_messages[-5:] if self.status_messages else []
        }
    
    def finish(self, success_message: Optional[str] = None):
        """
        Finish the multi-stage progress tracking.
        
        Args:
            success_message: Optional overall success message
        """
        if self.progress_bar:
            self.progress_bar.close()
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        if success_message:
            logger.info(f"{success_message} (total time: {self._format_time(total_time)})")
            print(f"\n✅ {success_message}")
        else:
            logger.info(f"Multi-stage process completed (total time: {self._format_time(total_time)})")
            print(f"\n✅ Process completed successfully!")
        
        # Display stage summary
        print(f"📊 Process Summary:")
        print(f"   Total time: {self._format_time(total_time)}")
        print(f"   Stages completed: {len(self.stage_times)}/{len(self.stages)}")
        
        for stage_info in self.stage_times:
            stage_time = self._format_time(stage_info['duration'])
            print(f"   • {stage_info['stage_name']}: {stage_time}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"


class RealTimeStatusDisplay:
    """
    Real-time status display for long-running operations.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize real-time status display.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.status_data = {}
        self.running = False
        self.display_thread = None
        self.last_update = 0
    
    def start(self):
        """Start the real-time status display."""
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        logger.info("Real-time status display started")
    
    def stop(self):
        """Stop the real-time status display."""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        logger.info("Real-time status display stopped")
    
    def update_status(self, key: str, value: Any):
        """
        Update a status value.
        
        Args:
            key: Status key
            value: Status value
        """
        self.status_data[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def _display_loop(self):
        """Main display loop running in separate thread."""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_update >= self.update_interval:
                    self._update_display()
                    self.last_update = current_time
                
                time.sleep(0.1)  # Small sleep to prevent high CPU usage
                
            except Exception as e:
                logger.error(f"Error in status display loop: {e}")
                break
    
    def _update_display(self):
        """Update the status display."""
        if not self.status_data:
            return
        
        # Clear previous lines (simple approach)
        # In a more sophisticated implementation, you might use curses or similar
        status_lines = []
        
        for key, data in self.status_data.items():
            value = data['value']
            age = time.time() - data['timestamp']
            
            # Format value based on type
            if isinstance(value, float):
                if 'rate' in key.lower() or 'fps' in key.lower():
                    formatted_value = f"{value:.1f}"
                elif 'percent' in key.lower() or '%' in key.lower():
                    formatted_value = f"{value:.1f}%"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            # Add age indicator for stale data
            if age > 10:
                formatted_value += " (stale)"
            
            status_lines.append(f"   {key}: {formatted_value}")
        
        # Print status (this is a simple implementation)
        # In production, you might want to use a more sophisticated display method
        if status_lines:
            print(f"\r📊 Status: {' | '.join(status_lines[:3])}", end='', flush=True)


class SuccessMessageManager:
    """
    Manager for displaying success messages and completion summaries.
    """
    
    def __init__(self):
        self.completion_messages = []
        self.statistics = {}
        self.start_time = None
    
    def start_operation(self, operation_name: str):
        """
        Start tracking an operation.
        
        Args:
            operation_name: Name of the operation
        """
        self.start_time = time.time()
        self.statistics['operation_name'] = operation_name
        logger.info(f"Started operation: {operation_name}")
    
    def add_completion_message(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Add a completion message for a stage.
        
        Args:
            stage: Stage name
            message: Completion message
            details: Optional additional details
        """
        completion_info = {
            'stage': stage,
            'message': message,
            'timestamp': time.time(),
            'details': details or {}
        }
        self.completion_messages.append(completion_info)
        
        print(f"✅ {stage}: {message}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
        
        logger.info(f"Stage completed - {stage}: {message}")
    
    def add_statistic(self, key: str, value: Any):
        """
        Add a statistic to track.
        
        Args:
            key: Statistic key
            value: Statistic value
        """
        self.statistics[key] = value
    
    def display_final_summary(self, output_path: Optional[str] = None):
        """
        Display final completion summary.
        
        Args:
            output_path: Path to output file if applicable
        """
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "="*60)
        print("🎉 OPERATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if self.statistics.get('operation_name'):
            print(f"Operation: {self.statistics['operation_name']}")
        
        print(f"Total time: {self._format_time(total_time)}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
            
            # Try to get output file size
            try:
                from pathlib import Path
                if Path(output_path).exists():
                    size_mb = Path(output_path).stat().st_size / (1024**2)
                    print(f"Output file size: {size_mb:.1f} MB")
            except:
                pass
        
        # Display key statistics
        if len(self.statistics) > 1:  # More than just operation_name
            print("\n📊 Summary Statistics:")
            for key, value in self.statistics.items():
                if key != 'operation_name':
                    if isinstance(value, float):
                        if 'time' in key.lower():
                            print(f"   {key}: {self._format_time(value)}")
                        elif 'rate' in key.lower() or 'fps' in key.lower():
                            print(f"   {key}: {value:.1f}")
                        else:
                            print(f"   {key}: {value:.2f}")
                    else:
                        print(f"   {key}: {value}")
        
        # Display stage completion summary
        if self.completion_messages:
            print(f"\n✅ Completed Stages ({len(self.completion_messages)}):")
            for msg in self.completion_messages:
                elapsed = msg['timestamp'] - self.start_time if self.start_time else 0
                print(f"   • {msg['stage']} ({self._format_time(elapsed)})")
        
        print("="*60)
        print("Thank you for using FaceSwap! 🚀")
        print("="*60)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"