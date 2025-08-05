#!/usr/bin/env python3
"""
Enhanced Batch Video Processing Script for AlphaPose with Configuration Support
This script processes multiple video files automatically using the video_demo functionality.
It can read settings from a configuration file for easier customization.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path
import argparse
import time
import configparser
from typing import List, Dict, Any

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from INI file"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    settings = {}
    
    # Paths
    settings['input_dir'] = config.get('PATHS', 'INPUT_DIR', fallback='../PD_rawvideos/raw_videos')
    settings['output_dir'] = config.get('PATHS', 'OUTPUT_DIR', fallback='../outputs/batch_results')
    
    # Detection
    settings['mode'] = config.get('DETECTION', 'MODE', fallback='normal')
    settings['confidence'] = config.getfloat('DETECTION', 'CONFIDENCE', fallback=0.05)
    settings['nms_threshold'] = config.getfloat('DETECTION', 'NMS_THRESHOLD', fallback=0.6)
    settings['detbatch'] = config.getint('DETECTION', 'DETECTION_BATCH_SIZE', fallback=1)
    settings['posebatch'] = config.getint('DETECTION', 'POSE_BATCH_SIZE', fallback=80)
    
    # Output
    settings['save_video'] = config.getboolean('OUTPUT', 'SAVE_VIDEO', fallback=True)
    settings['vis_fast'] = config.getboolean('OUTPUT', 'FAST_VISUALIZATION', fallback=False)
    settings['save_img'] = config.getboolean('OUTPUT', 'SAVE_IMAGES', fallback=False)
    
    # Processing
    settings['cpu'] = config.getboolean('PROCESSING', 'USE_CPU', fallback=False)
    settings['resume'] = config.getboolean('PROCESSING', 'RESUME', fallback=True)
    extensions_str = config.get('PROCESSING', 'VIDEO_EXTENSIONS', fallback='mp4,avi,mov,mkv')
    settings['extensions'] = [f'*.{ext.strip()}' for ext in extensions_str.split(',')]
    
    # Advanced
    settings['inp_dim'] = config.get('ADVANCED', 'INPUT_DIM', fallback='608')
    settings['format'] = config.get('ADVANCED', 'OUTPUT_FORMAT', fallback='coco')
    settings['profile'] = config.getboolean('ADVANCED', 'SHOW_PROFILE', fallback=False)
    
    return settings

def get_video_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """Get all video files from the input directory"""
    if extensions is None:
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    
    video_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        return []
    
    for ext in extensions:
        # Search recursively
        video_files.extend(input_path.rglob(ext))
    
    return sorted([str(f) for f in video_files])

def estimate_processing_time(video_path: str) -> float:
    """Estimate processing time based on video duration (rough estimate)"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Rough estimate: processing takes 2-5x video duration depending on hardware
        return duration * 3  # Conservative estimate
    except:
        return 0

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def process_single_video(video_path: str, output_dir: str, settings: Dict[str, Any]) -> bool:
    """Process a single video file"""
    video_name = Path(video_path).stem
    
    # Create individual folder for this video
    video_output_dir = os.path.join(output_dir, video_name)
    
    print(f"\n{'='*80}")
    print(f"Processing: {video_name}")
    print(f"Input: {video_path}")
    print(f"Output directory: {video_output_dir}")
    
    # Check if folder already exists and what's in it
    if os.path.exists(video_output_dir):
        existing_files = os.listdir(video_output_dir)
        print(f"Existing folder found with files: {existing_files}")
        
        # Check if this is already processed
        if check_already_processed(video_path, output_dir):
            print(f"Video already processed - files exist in: {video_output_dir}")
            return True
    
    # Estimate processing time
    estimated_time = estimate_processing_time(video_path)
    if estimated_time > 0:
        print(f"Estimated processing time: {format_time(estimated_time)}")
    
    print(f"{'='*80}")
    
    # Ensure video-specific output directory exists
    os.makedirs(video_output_dir, exist_ok=True)
    print(f"Using output directory: {video_output_dir}")
    
    # Build the command - use the video-specific directory
    cmd = [
        sys.executable, 'video_demo.py',
        '--video', video_path,
        '--outdir', video_output_dir,  # Use individual video folder
        '--mode', settings['mode'],
        '--conf', str(settings['confidence']),
        '--nms', str(settings['nms_threshold']),
        '--detbatch', str(settings['detbatch']),
        '--posebatch', str(settings['posebatch']),
        '--inp_dim', settings['inp_dim']
    ]
    
    # Add optional flags
    if settings['save_video']:
        cmd.append('--save_video')
    if settings['vis_fast']:
        cmd.append('--vis_fast')
    if settings['save_img']:
        cmd.append('--save_img')
    if settings['cpu']:
        cmd.append('--cpu')
    if settings['profile']:
        cmd.append('--profile')
    if 'format' in settings and settings['format']:
        cmd.extend(['--format', settings['format']])
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting processing...")
    
    # Execute the command
    start_time = time.time()
    try:
        # Run with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, cwd=os.path.dirname(__file__))
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        end_time = time.time()
        
        if process.returncode == 0:
            processing_time = end_time - start_time
            print(f"\n Successfully processed {video_name} in {format_time(processing_time)}")
            return True
        else:
            print(f"\n Failed to process {video_name} (exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\n Error processing {video_name}: {str(e)}")
        return False

def check_already_processed(video_path: str, output_dir: str) -> bool:
    """Check if video has already been processed"""
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    
    # Check for files in the individual video folder
    # The actual output video is named AlphaPose_[video_name].mp4
    expected_output_1 = os.path.join(video_output_dir, 'alphapose_output.mp4')  # New format
    expected_output_2 = os.path.join(video_output_dir, f'AlphaPose_{video_name}.mp4')  # Existing format
    expected_json = os.path.join(video_output_dir, 'alphapose-results.json')
    
    # Check if folder exists and contains the required files
    if not os.path.exists(video_output_dir):
        return False
    
    # Check for JSON file (required)
    if not os.path.exists(expected_json):
        return False
    
    # Check for either video format (old or new)
    return os.path.exists(expected_output_1) or os.path.exists(expected_output_2)

def main():
    parser = argparse.ArgumentParser(description='Batch process videos with AlphaPose')
    parser.add_argument('--config', '-c', default='batch_config.ini',
                       help='Configuration file path')
    parser.add_argument('--input_dir', '-i',
                       help='Directory containing input videos (overrides config)')
    parser.add_argument('--output_dir', '-o',
                       help='Directory to save processed videos (overrides config)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be processed without actually processing')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU processing (overrides config)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable detailed profiling information during processing')
    parser.add_argument('--vis_fast', action='store_true',
                       help='Enable fast visualization for faster processing (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    settings = {}
    if os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        settings = load_config(args.config)
    else:
        print(f"Configuration file not found: {args.config}")
        print("Using default settings...")
        settings = {
            'input_dir': '../PD_rawvideos/raw_videos',
            'output_dir': '../outputs/batch_results',
            'mode': 'normal',
            'confidence': 0.05,
            'nms_threshold': 0.6,
            'detbatch': 1,
            'posebatch': 80,
            'save_video': True,
            'vis_fast': False,
            'save_img': False,
            'cpu': True,  # Default to CPU for better compatibility
            'resume': True,
            'extensions': ['*.mp4', '*.avi', '*.mov', '*.mkv'],
            'inp_dim': '608',
            'format': 'coco',
            'profile': False
        }
    
    # Override with command line arguments
    if args.input_dir:
        settings['input_dir'] = args.input_dir
    if args.output_dir:
        settings['output_dir'] = args.output_dir
    if args.cpu:
        settings['cpu'] = True
    if args.profile:
        settings['profile'] = True
    if args.vis_fast:
        settings['vis_fast'] = True
    
    # Convert relative paths to absolute
    settings['input_dir'] = os.path.abspath(settings['input_dir'])
    settings['output_dir'] = os.path.abspath(settings['output_dir'])
    
    # Validate input directory
    if not os.path.exists(settings['input_dir']):
        print(f"ERROR: Input directory does not exist: {settings['input_dir']}")
        return
    
    # Get all video files
    print(f"\nSearching for videos in: {settings['input_dir']}")
    video_files = get_video_files(settings['input_dir'], settings['extensions'])
    
    if not video_files:
        print(f"ERROR: No video files found in {settings['input_dir']}")
        print(f"Searched for extensions: {settings['extensions']}")
        return
    
    print(f"\nFound {len(video_files)} video files:")
    total_estimated_time = 0
    for i, video in enumerate(video_files, 1):
        rel_path = os.path.relpath(video, settings['input_dir'])
        estimated = estimate_processing_time(video)
        total_estimated_time += estimated
        status = ""
        
        if settings['resume'] and check_already_processed(video, settings['output_dir']):
            status = " (already processed - will skip)"
        
        print(f"  {i:2d}. {rel_path}{status}")
        if estimated > 0:
            print(f"      Estimated time: {format_time(estimated)}")
    
    if total_estimated_time > 0:
        print(f"\nTotal estimated processing time: {format_time(total_estimated_time)}")
    
    # Show settings
    print(f"\nProcessing settings:")
    print(f"  Mode: {settings['mode']}")
    print(f"  Confidence: {settings['confidence']}")
    print(f"  NMS threshold: {settings['nms_threshold']}")
    print(f"  Detection batch: {settings['detbatch']}")
    print(f"  Pose batch: {settings['posebatch']}")
    print(f"  Save video: {settings['save_video']}")
    print(f"  Fast visualization: {settings['vis_fast']}")
    print(f"  Use CPU: {settings['cpu']}")
    print(f"  Resume: {settings['resume']}")
    
    if args.dry_run:
        print("\nDRY RUN - No videos will be processed")
        return
    
    # Confirm before processing
    print(f"\nOutput directory: {settings['output_dir']}")
    response = input("\nProceed with batch processing? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Processing cancelled.")
        return
    
    # Process videos
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\n{'='*80}")
    print("STARTING BATCH PROCESSING")
    print(f"{'='*80}")
    print(f"Total videos to process: {len(video_files)}")
    print(f"{'='*80}")
    
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[PROGRESS: {i}/{len(video_files)}] ({i/len(video_files)*100:.1f}% complete)")
        
        # Check if already processed (resume functionality)
        if settings['resume'] and check_already_processed(video_path, settings['output_dir']):
            video_name = Path(video_path).stem
            print(f"SKIPPING: {video_name} (already processed)")
            skipped += 1
            continue
        
        # Process the video
        if process_single_video(video_path, settings['output_dir'], settings):
            successful += 1
            print(f"\n[SUCCESS] Completed {successful}/{len(video_files)} videos")
        else:
            failed += 1
            print(f"\n[FAILED] {failed} failures so far")
            print(f"ERROR: Failed to process {os.path.basename(video_path)}")
            
            # Ask if user wants to continue after failure
            if i < len(video_files):
                response = input("Continue with remaining videos? (Y/n): ")
                if response.lower() in ['n', 'no']:
                    print("Processing stopped by user.")
                    break
        
        # Show current progress summary
        elapsed_time = time.time() - total_start_time
        processed = successful + failed
        remaining = len(video_files) - i
        
        if processed > 0:
            avg_time_per_video = elapsed_time / processed
            estimated_remaining_time = avg_time_per_video * remaining
            
            print(f"\n[PROGRESS UPDATE]")
            print(f"  Processed: {processed}/{len(video_files)} ({processed/len(video_files)*100:.1f}%)")
            print(f"  Success: {successful} | Failed: {failed} | Skipped: {skipped}")
            print(f"  Time elapsed: {format_time(elapsed_time)}")
            print(f"  Estimated remaining: {format_time(estimated_remaining_time)}")
        
        print(f"{'='*80}")
    
    total_end_time = time.time()
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos found: {len(video_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total processing time: {format_time(total_end_time - total_start_time)}")
    print(f"Output directory: {settings['output_dir']}")
    
    if successful > 0:
        print(f"\nSUCCESS: Batch processing completed!")
        print(f"Check your results in: {settings['output_dir']}")
    
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
