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
from typing import List, Dict, Any, Optional
from tqdm import tqdm

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

def extract_frames(video_path: str, output_root: str, stride: int = 1, overwrite: bool = False, limit: Optional[int] = None, img_ext: str = 'jpg') -> int:
    """Extract frames from a video.

    Args:
        video_path: Path to video file.
        output_root: Root output directory where a subfolder '<video_name>/frames' will be created.
        stride: Save every Nth frame (1 = all frames).
        overwrite: If False and frames folder exists & not empty, skip extraction.
        limit: Optional max number of frames to save (after stride filtering).
        img_ext: 'jpg' or 'png'.
    Returns:
        Number of frames saved.
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return 0

    video_name = Path(video_path).stem
    frames_dir = os.path.join(output_root, video_name, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Skip if already extracted
    if not overwrite and any(Path(frames_dir).glob(f'frame_*.{img_ext}')):
        print(f"Frames already exist for {video_name}, skipping (use --overwrite_frames to force).")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    saved = 0
    idx = 0
    pbar = None
    try:
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=total, desc=f"Extracting {video_name}", unit="frame")
    except Exception:
        pass

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            out_path = os.path.join(frames_dir, f"frame_{idx:06d}.{img_ext}")
            cv2.imwrite(out_path, frame)
            saved += 1
            if limit and saved >= limit:
                break
        idx += 1
        if pbar:
            pbar.update(1)

    cap.release()
    if pbar:
        pbar.close()
    print(f"Saved {saved} frame(s) to {frames_dir}")
    return saved

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
        print(f"Existing folder found with {len(existing_files)} files: {existing_files}")
        
        # Detailed check of what exists
        expected_output = os.path.join(video_output_dir, f"{video_name}.mp4")
        expected_json = os.path.join(video_output_dir, f"{video_name}.json")

        video_exists = os.path.exists(expected_output)
        json_exists = os.path.exists(expected_json)
        
        print(f"File status check:")
        print(f"  Video file: {'EXISTS' if video_exists else 'MISSING'}")
        print(f"  JSON file:  {'EXISTS' if json_exists else 'MISSING'}")
        
        # Check if this is already completely processed
        if video_exists and json_exists:
            print(f"Video already completely processed - files exist in: {video_output_dir}")
            return True
        elif video_exists or json_exists:
            print(f"Incomplete processing detected - will reprocess to ensure both files exist")
        else:
            print(f"No output files found - will process from scratch")
    else:
        print(f"No existing output folder - will create and process from scratch")
    
    # Estimate processing time
    estimated_time = estimate_processing_time(video_path)
    if estimated_time > 0:
        print(f"Estimated processing time: {format_time(estimated_time)}")
    
    print(f"{'='*80}")
    
    # Ensure main output directory exists (let video_demo.py create the video-specific folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using output directory: {output_dir} (video_demo.py will create subfolder: {video_name})")
    
    # Build the command - use the main output directory, let video_demo.py create the video subfolder
    cmd = [
        sys.executable, 'video_demo.py',
        '--video', video_path,
        '--outdir', output_dir,  # Use main output directory, not video-specific folder
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
        # Set environment variables for CPU-only processing to avoid CUDA issues
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
        env['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
        
        # Run with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, cwd=os.path.dirname(__file__), env=env)
        
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
    
    # Check if folder exists and contains the required files
    if not os.path.exists(video_output_dir):
        return False
    
    # Check for files in the individual video folder
    expected_output = os.path.join(video_output_dir, f"{video_name}.mp4")
    expected_json = os.path.join(video_output_dir, f"{video_name}.json")
    # Check what files exist
    video_exists = os.path.exists(expected_output)
    json_exists = os.path.exists(expected_json)
    
    # Print detailed status
    if video_exists and json_exists:
        print(f"Complete: Both video and JSON files exist for {video_name}")
        return True
    elif video_exists and not json_exists:
        print(f"Incomplete: Video exists but JSON missing for {video_name}")
        return False
    elif not video_exists and json_exists:
        print(f"Incomplete: JSON exists but video missing for {video_name}")
        return False
    else:
        print(f"Missing: No output files found for {video_name}")
        return False

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
    parser.add_argument('--extract_frames', action='store_true',
                       help='Extract frames for each processed (or all) video')
    parser.add_argument('--frame_stride', type=int, default=1,
                       help='Save every Nth frame (default=1)')
    parser.add_argument('--frame_limit', type=int, default=None,
                       help='Optional max number of frames to save per video (after stride)')
    parser.add_argument('--overwrite_frames', action='store_true',
                       help='Overwrite existing extracted frames if present')
    parser.add_argument('--extract_only', action='store_true',
                       help='Only extract frames without running pose estimation')
    parser.add_argument('--frame_ext', default='jpg', choices=['jpg','png'],
                       help='Image extension for saved frames')
    
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
            'mode': 'fast',  # Use fast mode for better performance
            'confidence': 0.05,
            'nms_threshold': 0.6,
            'detbatch': 1,  # Keep detection batch small for memory
            'posebatch': 1,  # Reduce pose batch for memory constraints
            'save_video': True,
            'vis_fast': True,  # Default to fast visualization for better performance
            'save_img': False,
            'cpu': True,  # Default to CPU for better compatibility
            'resume': True,
            'extensions': ['*.mp4', '*.avi', '*.mov', '*.mkv'],
            'inp_dim': '416',  # Reduced input dimensions for lower memory
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
    print("Checking existing output files...")
    print(f"{'='*60}")
    
    total_estimated_time = 0
    already_processed_count = 0
    incomplete_count = 0
    missing_count = 0
    
    for i, video in enumerate(video_files, 1):
        rel_path = os.path.relpath(video, settings['input_dir'])
        estimated = estimate_processing_time(video)
        total_estimated_time += estimated
        
        # Check processing status
        video_name = Path(video).stem
        video_output_dir = os.path.join(settings['output_dir'], video_name)
        
        if os.path.exists(video_output_dir):
            # Check what files exist
            expected_output = os.path.join(video_output_dir, f"{video_name}.mp4")
            video_name = os.path.basename(video_output_dir)
            json_filename = f"{video_name}.json"
            expected_json = os.path.join(video_output_dir, json_filename)
            
            video_exists = os.path.exists(expected_output)
            json_exists = os.path.exists(expected_json)
            
            if video_exists and json_exists:
                status = " [COMPLETE]"
                already_processed_count += 1
            elif video_exists or json_exists:
                status = " [INCOMPLETE]"
                incomplete_count += 1
            else:
                status = " [MISSING]"
                missing_count += 1
        else:
            status = " [MISSING]"
            missing_count += 1
        
        print(f"  {i:2d}. {rel_path}{status}")
        if estimated > 0:
            print(f"      Estimated time: {format_time(estimated)}")
    
    print(f"{'='*60}")
    print(f"Status Summary:")
    print(f"  [COMPLETE] (will skip): {already_processed_count}")
    print(f"  [INCOMPLETE] (will reprocess): {incomplete_count}")
    print(f"  [MISSING] (will process): {missing_count}")
    print(f"  Total videos to process: {incomplete_count + missing_count}")
    
    if settings['resume']:
        videos_to_process = incomplete_count + missing_count
        if videos_to_process == 0:
            print(f"\nAll videos already processed! No work needed.")
            return
        else:
            print(f"\nResume mode enabled - will skip {already_processed_count} completed videos")
    
    if total_estimated_time > 0:
        # Adjust time estimate for resume mode
        if settings['resume']:
            estimated_time_for_remaining = total_estimated_time * (videos_to_process / len(video_files))
            print(f"Estimated processing time for remaining videos: {format_time(estimated_time_for_remaining)}")
        else:
            print(f"Total estimated processing time: {format_time(total_estimated_time)}")
    
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
    if not args.extract_only:
        response = input("\nProceed with batch processing? (Y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Processing cancelled.")
            return
    else:
        response = input("\nProceed with frame extraction only? (Y/N): ")
        if response.lower() not in ['y','yes']:
            print("Extraction cancelled.")
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
    
    # Create progress bar for batch processing
    video_progress = tqdm(video_files, desc="Batch Processing", unit="video", 
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, video_path in enumerate(video_progress, 1):
        video_name = Path(video_path).stem
        video_progress.set_description(f"Processing: {video_name}")
        
        print(f"\n[PROGRESS: {i}/{len(video_files)}] ({i/len(video_files)*100:.1f}% complete)")

        # Frame extraction only mode
        if args.extract_only:
            extract_frames(video_path, settings['output_dir'], stride=args.frame_stride,
                           overwrite=args.overwrite_frames, limit=args.frame_limit, img_ext=args.frame_ext)
            skipped += 1  # treat as skipped for pose estimation
            video_progress.set_postfix({"Frames": "extracted", "SkippedPE": skipped})
            continue

        # Check if already processed (resume functionality)
        if settings['resume'] and check_already_processed(video_path, settings['output_dir']):
            print(f"SKIPPING POSE: {video_name} (already processed)")
            skipped += 1
            # Still extract frames if requested
            if args.extract_frames:
                extract_frames(video_path, settings['output_dir'], stride=args.frame_stride,
                               overwrite=args.overwrite_frames, limit=args.frame_limit, img_ext=args.frame_ext)
            video_progress.set_postfix({"Success": successful, "Failed": failed, "Skipped": skipped})
            continue

        # Process the video (pose estimation)
        try:
            pose_ok = process_single_video(video_path, settings['output_dir'], settings)
            if pose_ok:
                successful += 1
                print(f"\n[SUCCESS] Completed {successful}/{len(video_files)} videos")
                # Optional frame extraction after successful processing
                if args.extract_frames:
                    extract_frames(video_path, settings['output_dir'], stride=args.frame_stride,
                                   overwrite=args.overwrite_frames, limit=args.frame_limit, img_ext=args.frame_ext)
            else:
                failed += 1
                print(f"\n[FAILED] {failed} failures so far")
                print(f"ERROR: Failed to process {os.path.basename(video_path)}")
                
            video_progress.set_postfix({"Success": successful, "Failed": failed, "Skipped": skipped})

            if not pose_ok and i < len(video_files):
                response = input("Continue with remaining videos? (Y/n): ")
                if response.lower() in ['n', 'no']:
                    print("Processing stopped by user.")
                    break
        except MemoryError as e:
            failed += 1
            print(f"\n[MEMORY ERROR] Failed to process {os.path.basename(video_path)}: {str(e)}")
            print("This might be due to insufficient system memory. Try:")
            print("1. Reducing batch sizes in configuration")
            print("2. Processing videos one at a time")
            print("3. Restarting the script to free memory")
            video_progress.set_postfix({"Success": successful, "Failed": failed, "Skipped": skipped})
            response = input("Continue with remaining videos? (Y/n): ")
            if response.lower() in ['n', 'no']:
                print("Processing stopped by user.")
                break
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] Unexpected error processing {os.path.basename(video_path)}: {str(e)}")
            video_progress.set_postfix({"Success": successful, "Failed": failed, "Skipped": skipped})
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
    
    # Close the progress bar
    video_progress.close()
    
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
