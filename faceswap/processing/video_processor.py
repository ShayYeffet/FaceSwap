"""
Video processing module for face swapping application.

This module handles video frame extraction, processing, and reconstruction
while preserving original video properties like fps, resolution, and audio.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, Generator, Dict, Any, List
from pathlib import Path
import tempfile
import subprocess
import os

from ..utils.logger import get_logger
from ..utils.file_manager import FileManager
from ..utils.progress_tracker import ProgressTracker
from .face_detector import FaceDetector, FaceData
from .face_swapper import FaceSwapper, SwapResult

logger = get_logger(__name__)


class VideoProcessor:
    """
    Handles video frame extraction, processing, and reconstruction.
    
    Supports common video formats and preserves original video properties
    including fps, resolution, and audio tracks.
    """
    
    def __init__(self, 
                 face_detector: Optional[FaceDetector] = None,
                 face_swapper: Optional[FaceSwapper] = None,
                 temp_dir: Optional[str] = None):
        """
        Initialize the video processor.
        
        Args:
            face_detector: FaceDetector instance for face processing
            face_swapper: FaceSwapper instance for face swapping operations
            temp_dir: Temporary directory for frame storage
        """
        self.face_detector = face_detector or FaceDetector()
        self.face_swapper = face_swapper
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="faceswap_")
        self.video_properties = {}
        
        # Ensure temp directory exists
        FileManager.ensure_directory_exists(self.temp_dir)
        logger.info(f"VideoProcessor initialized with temp dir: {self.temp_dir}")
    
    def get_video_properties(self, video_path: str) -> Dict[str, Any]:
        """
        Extract video properties using OpenCV.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Dictionary containing video properties
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            properties = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            
            # Convert fourcc to string
            fourcc_str = "".join([chr((properties['fourcc'] >> 8 * i) & 0xFF) for i in range(4)])
            properties['fourcc_str'] = fourcc_str
            
            logger.info(f"Video properties: {properties['width']}x{properties['height']}, "
                       f"{properties['fps']:.2f} fps, {properties['frame_count']} frames, "
                       f"{properties['duration']:.2f}s")
            
            return properties
            
        except Exception as e:
            logger.error(f"Error getting video properties: {e}")
            raise
    
    def extract_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video file as a generator.
        
        Args:
            video_path: Path to the input video file
            
        Yields:
            Tuple of (frame_number, frame_image)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                yield frame_number, frame
                frame_number += 1
            
            cap.release()
            logger.info(f"Extracted {frame_number} frames from video")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def extract_frames_batch(self, 
                           video_path: str, 
                           batch_size: int = 32) -> Generator[Tuple[int, list], None, None]:
        """
        Extract frames in batches for efficient processing.
        
        Args:
            video_path: Path to the input video file
            batch_size: Number of frames per batch
            
        Yields:
            Tuple of (start_frame_number, list_of_frames)
        """
        try:
            batch = []
            start_frame = 0
            
            for frame_number, frame in self.extract_frames(video_path):
                if len(batch) == 0:
                    start_frame = frame_number
                
                batch.append(frame)
                
                if len(batch) >= batch_size:
                    yield start_frame, batch
                    batch = []
            
            # Yield remaining frames
            if batch:
                yield start_frame, batch
                
        except Exception as e:
            logger.error(f"Error extracting frame batches: {e}")
            raise
    
    def save_frames_to_temp(self, 
                          video_path: str, 
                          frame_format: str = "frame_%06d.jpg") -> str:
        """
        Save all video frames to temporary directory.
        
        Args:
            video_path: Path to the input video file
            frame_format: Format string for frame filenames
            
        Returns:
            Path to the temporary directory containing frames
        """
        try:
            frames_dir = os.path.join(self.temp_dir, "frames")
            FileManager.ensure_directory_exists(frames_dir)
            
            frame_count = 0
            for frame_number, frame in self.extract_frames(video_path):
                frame_path = os.path.join(frames_dir, frame_format % frame_number)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            
            logger.info(f"Saved {frame_count} frames to {frames_dir}")
            return frames_dir
            
        except Exception as e:
            logger.error(f"Error saving frames to temp: {e}")
            raise
    
    def reconstruct_video(self, 
                         frames_dir: str, 
                         output_path: str, 
                         video_properties: Dict[str, Any],
                         frame_format: str = "frame_%06d.jpg") -> bool:
        """
        Reconstruct video from processed frames.
        
        Args:
            frames_dir: Directory containing processed frames
            output_path: Path for the output video file
            video_properties: Original video properties
            frame_format: Format string for frame filenames
            
        Returns:
            True if reconstruction successful, False otherwise
        """
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for compatibility
            fps = video_properties['fps']
            width = video_properties['width']
            height = video_properties['height']
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Cannot create video writer for: {output_path}")
            
            # Get list of frame files
            frames_path = Path(frames_dir)
            frame_files = sorted([f for f in frames_path.iterdir() 
                                if f.suffix.lower() in ['.jpg', '.png']])
            
            if not frame_files:
                raise ValueError(f"No frame files found in: {frames_dir}")
            
            # Write frames to video
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_file}")
                    continue
                
                # Resize frame if necessary
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
            
            out.release()
            
            logger.info(f"Video reconstruction complete: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return False
    
    def reconstruct_video_with_audio(self, 
                                   video_without_audio: str,
                                   original_video: str,
                                   output_path: str) -> bool:
        """
        Combine processed video with original audio using FFmpeg.
        
        Args:
            video_without_audio: Path to processed video without audio
            original_video: Path to original video with audio
            output_path: Path for final output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if FFmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], 
                             capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("FFmpeg not found. Output video will not have audio.")
                # Just copy the video without audio
                import shutil
                shutil.copy2(video_without_audio, output_path)
                return True
            
            # Use FFmpeg to combine video and audio
            cmd = [
                'ffmpeg',
                '-i', video_without_audio,  # Video input
                '-i', original_video,       # Audio input
                '-c:v', 'copy',            # Copy video codec
                '-c:a', 'aac',             # Use AAC audio codec
                '-map', '0:v:0',           # Map video from first input
                '-map', '1:a:0',           # Map audio from second input
                '-shortest',               # End when shortest stream ends
                '-y',                      # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully combined video with audio: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                # Fallback: copy video without audio
                import shutil
                shutil.copy2(video_without_audio, output_path)
                return True
                
        except Exception as e:
            logger.error(f"Error combining video with audio: {e}")
            return False
    
    def process_video_frames(self, 
                           video_path: str,
                           output_path: str,
                           frame_processor_func,
                           batch_size: int = 16,
                           preserve_audio: bool = True) -> bool:
        """
        Process video frames using a custom processing function.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            frame_processor_func: Function to process frames (takes frame, returns processed frame)
            batch_size: Number of frames to process in batch
            preserve_audio: Whether to preserve original audio
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Get video properties
            self.video_properties = self.get_video_properties(video_path)
            
            # Create temporary directories
            temp_input_dir = os.path.join(self.temp_dir, "input_frames")
            temp_output_dir = os.path.join(self.temp_dir, "output_frames")
            FileManager.ensure_directory_exists(temp_input_dir)
            FileManager.ensure_directory_exists(temp_output_dir)
            
            # Initialize progress tracker
            total_frames = self.video_properties['frame_count']
            progress = ProgressTracker(total_frames, "Processing video frames")
            
            # Process frames in batches
            processed_count = 0
            
            for start_frame, frame_batch in self.extract_frames_batch(video_path, batch_size):
                processed_frames = []
                
                # Process each frame in the batch
                for i, frame in enumerate(frame_batch):
                    try:
                        processed_frame = frame_processor_func(frame)
                        processed_frames.append(processed_frame)
                    except Exception as e:
                        logger.warning(f"Error processing frame {start_frame + i}: {e}")
                        # Use original frame as fallback
                        processed_frames.append(frame)
                
                # Save processed frames
                for i, processed_frame in enumerate(processed_frames):
                    frame_number = start_frame + i
                    frame_path = os.path.join(temp_output_dir, f"frame_{frame_number:06d}.jpg")
                    cv2.imwrite(frame_path, processed_frame)
                    
                    processed_count += 1
                    progress.update(processed_count, f"Processed frame {processed_count}/{total_frames}")
            
            progress.complete("Frame processing complete")
            
            # Reconstruct video without audio
            temp_video_path = os.path.join(self.temp_dir, "temp_video.mp4")
            if not self.reconstruct_video(temp_output_dir, temp_video_path, self.video_properties):
                return False
            
            # Combine with audio if requested
            if preserve_audio:
                success = self.reconstruct_video_with_audio(temp_video_path, video_path, output_path)
            else:
                import shutil
                shutil.copy2(temp_video_path, output_path)
                success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()
    
    def get_frame_at_time(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a specific frame at given timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            
        Returns:
            Frame image or None if extraction fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            # Set position to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"Error extracting frame at timestamp {timestamp}: {e}")
            return None
    
    def get_frame_at_index(self, video_path: str, frame_index: int) -> Optional[np.ndarray]:
        """
        Extract a specific frame by index.
        
        Args:
            video_path: Path to video file
            frame_index: Frame number (0-based)
            
        Returns:
            Frame image or None if extraction fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            # Set position to frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            logger.error(f"Error extracting frame at index {frame_index}: {e}")
            return None
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files and directories."""
        try:
            FileManager.clean_temp_files(self.temp_dir)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
    
    def process_video(self, 
                     video_path: str,
                     output_path: str,
                     quality: str = 'balanced',
                     preview_frames: Optional[int] = None,
                     progress_callback=None) -> Dict:
        """
        Main video processing method expected by the application.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            quality: Processing quality ('fast', 'balanced', 'high')
            preview_frames: Number of frames to process (None for all)
            progress_callback: Progress callback function
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Get video properties
            self.video_properties = self.get_video_properties(video_path)
            total_frames = self.video_properties.get('frame_count', 0)
            
            # Limit frames if preview mode
            frames_to_process = min(preview_frames, total_frames) if preview_frames else total_frames
            
            if progress_callback:
                progress_callback(0, frames_to_process, "Starting video processing...")
            
            # Implement actual face swapping video processing
            import cv2
            import torch
            import numpy as np
            import tempfile
            import os
            
            logger.info(f"Starting real face swapping: {video_path} -> {output_path}")
            
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Could not create output video file: {output_path}")
            
            processed_frames = 0
            swap_success_count = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Stop if we've processed enough frames for preview
                    if frames_to_process and processed_frames >= frames_to_process:
                        break
                    
                    processed_frame = frame.copy()
                    
                    # Detect faces in the frame
                    if self.face_detector:
                        faces = self.face_detector.detect_faces(frame)
                        
                        if faces and self.face_swapper:
                            # Try to swap faces
                            try:
                                swap_result = self.face_swapper.swap_faces_in_frame(frame, faces)
                                if swap_result.success:
                                    processed_frame = swap_result.swapped_frame
                                    swap_success_count += 1
                            except Exception as e:
                                logger.warning(f"Face swap failed for frame {processed_frames}: {e}")
                    
                    # Write the processed frame
                    out.write(processed_frame)
                    processed_frames += 1
                    
                    if progress_callback:
                        progress_callback(processed_frames, frames_to_process or total_frames, 
                                        f"Processing frame {processed_frames}/{frames_to_process or total_frames}")
                
            finally:
                cap.release()
                out.release()
            
            swap_success_rate = swap_success_count / max(processed_frames, 1)
            logger.info(f"Video processing completed. Processed {processed_frames} frames, "
                       f"swap success rate: {swap_success_rate:.1%}")
            
            return {
                'frames_processed': processed_frames,
                'processing_fps': processed_frames / max(1, frames_to_process or total_frames) * fps,
                'swap_success_rate': swap_success_rate
            }
            
            return {
                'frames_processed': frames_to_process,
                'processing_fps': 25.0,  # Mock FPS
                'swap_success_rate': 0.95  # Mock success rate
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise

    def process_video_with_face_swap(self, 
                                   input_video_path: str,
                                   output_video_path: str,
                                   decoder_type: str = 'target',
                                   batch_size: int = 16,
                                   preserve_audio: bool = True) -> bool:
        """
        Process video with face swapping using the trained model.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path for output video
            decoder_type: Which decoder to use ('source' or 'target')
            batch_size: Number of frames to process in batch
            preserve_audio: Whether to preserve original audio
            
        Returns:
            True if processing successful, False otherwise
        """
        if self.face_swapper is None:
            logger.error("FaceSwapper not initialized. Cannot perform face swapping.")
            return False
        
        try:
            # Get video properties
            self.video_properties = self.get_video_properties(input_video_path)
            
            # Create temporary directories
            temp_output_dir = os.path.join(self.temp_dir, "swapped_frames")
            FileManager.ensure_directory_exists(temp_output_dir)
            
            # Initialize progress tracker
            total_frames = self.video_properties['frame_count']
            progress = ProgressTracker(total_frames, "Processing video with face swap")
            
            # Process frames in batches
            processed_count = 0
            all_results = []
            
            for start_frame, frame_batch in self.extract_frames_batch(input_video_path, batch_size):
                # Process batch with face swapping
                batch_results = self.face_swapper.process_frame_batch(frame_batch, decoder_type)
                all_results.extend(batch_results)
                
                # Save processed frames
                for i, result in enumerate(batch_results):
                    frame_number = start_frame + i
                    frame_path = os.path.join(temp_output_dir, f"frame_{frame_number:06d}.jpg")
                    
                    # Use swapped frame if successful, otherwise use original
                    frame_to_save = result.swapped_frame
                    cv2.imwrite(frame_path, frame_to_save)
                    
                    processed_count += 1
                    status = "swapped" if result.success else "original"
                    progress.update(processed_count, 
                                  f"Processed frame {processed_count}/{total_frames} ({status})")
            
            progress.complete("Face swap processing complete")
            
            # Log processing statistics
            stats = self.face_swapper.get_swap_statistics(all_results)
            logger.info(f"Face swap statistics: {stats['successful_swaps']}/{stats['total_frames']} "
                       f"successful swaps ({stats['success_rate']:.1%} success rate)")
            
            # Reconstruct video without audio
            temp_video_path = os.path.join(self.temp_dir, "temp_swapped_video.mp4")
            if not self.reconstruct_video(temp_output_dir, temp_video_path, self.video_properties):
                return False
            
            # Combine with audio if requested
            if preserve_audio:
                success = self.reconstruct_video_with_audio(temp_video_path, input_video_path, output_video_path)
            else:
                import shutil
                shutil.copy2(temp_video_path, output_video_path)
                success = True
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing video with face swap: {e}")
            return False
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()
    
    def preview_face_swap(self, 
                         video_path: str, 
                         frame_indices: List[int],
                         decoder_type: str = 'target') -> List[Tuple[np.ndarray, SwapResult]]:
        """
        Preview face swapping on specific frames.
        
        Args:
            video_path: Path to input video
            frame_indices: List of frame indices to preview
            decoder_type: Which decoder to use
            
        Returns:
            List of tuples (original_frame, swap_result)
        """
        if self.face_swapper is None:
            logger.error("FaceSwapper not initialized. Cannot preview face swapping.")
            return []
        
        try:
            results = []
            
            for frame_idx in frame_indices:
                # Extract specific frame
                original_frame = self.get_frame_at_index(video_path, frame_idx)
                
                if original_frame is None:
                    logger.warning(f"Could not extract frame {frame_idx}")
                    continue
                
                # Perform face swap
                swap_result = self.face_swapper.swap_face_in_frame(original_frame, decoder_type)
                results.append((original_frame, swap_result))
                
                logger.info(f"Preview frame {frame_idx}: {'Success' if swap_result.success else 'Failed'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error previewing face swap: {e}")
            return []
    
    def detect_faces_in_video_sample(self, 
                                   video_path: str, 
                                   sample_frames: int = 10) -> Dict[str, Any]:
        """
        Analyze face detection quality in a sample of video frames.
        
        Args:
            video_path: Path to input video
            sample_frames: Number of frames to sample
            
        Returns:
            Dictionary with face detection statistics
        """
        try:
            # Get video properties
            properties = self.get_video_properties(video_path)
            total_frames = properties['frame_count']
            
            # Calculate frame indices to sample
            if sample_frames >= total_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames // sample_frames
                frame_indices = list(range(0, total_frames, step))[:sample_frames]
            
            # Analyze frames
            detection_results = []
            
            for frame_idx in frame_indices:
                frame = self.get_frame_at_index(video_path, frame_idx)
                
                if frame is None:
                    continue
                
                # Detect faces
                face_data = self.face_detector.process_image(frame)
                
                detection_results.append({
                    'frame_index': frame_idx,
                    'face_detected': face_data is not None,
                    'confidence': face_data.confidence if face_data else 0.0,
                    'bbox_area': (face_data.bbox.width * face_data.bbox.height) if face_data else 0
                })
            
            # Calculate statistics
            total_analyzed = len(detection_results)
            faces_detected = sum(1 for r in detection_results if r['face_detected'])
            confidences = [r['confidence'] for r in detection_results if r['face_detected']]
            
            stats = {
                'total_frames_analyzed': total_analyzed,
                'faces_detected': faces_detected,
                'detection_rate': faces_detected / total_analyzed if total_analyzed > 0 else 0.0,
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'min_confidence': min(confidences) if confidences else 0.0,
                'max_confidence': max(confidences) if confidences else 0.0,
                'frame_results': detection_results
            }
            
            logger.info(f"Face detection analysis: {faces_detected}/{total_analyzed} frames "
                       f"({stats['detection_rate']:.1%} detection rate)")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing face detection in video: {e}")
            return {}
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup_temp_files()