#!/usr/bin/env python3
"""
Batch Video Processing Script for AlphaPose
This script processes multiple video files automatically using the video_demo functionality.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path
import argparse
import time

def get_video_files(input_dir, extensions=None):
    """Get all video files from the input directory"""
    if extensions is None:
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    
    video_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(video_files)

def process_single_video(video_path, output_dir, additional_args=None):
    """Process a single video file"""
    if additional_args is None:
        additional_args = []
    
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"Input: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command
    cmd = [
        sys.executable, 'video_demo.py',
        '--video', video_path,
        '--outdir', output_dir,
        '--save_video'  # Always save video output for batch processing
    ]
    
    # Add additional arguments
    cmd.extend(additional_args)
    
    print(f"Command: {' '.join(cmd)}")
    
    # Execute the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"SUCCESS: Processed {video_name} in {end_time - start_time:.2f} seconds")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"FAILED: Could not process {video_name}")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        print(f"ERROR: Exception while processing {video_name}: {str(e)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Batch process videos with AlphaPose')
    parser.add_argument('--input_dir', '-i', required=True,
                       help='Directory containing input videos')
    parser.add_argument('--output_dir', '-o', required=True,
                       help='Directory to save processed videos')
    parser.add_argument('--mode', default='normal',
                       help='Detection mode: fast/normal/accurate')
    parser.add_argument('--conf', type=float, default=0.05,
                       help='Confidence threshold for detection')
    parser.add_argument('--nms', type=float, default=0.6,
                       help='NMS threshold')
    parser.add_argument('--detbatch', type=int, default=1,
                       help='Detection batch size')
    parser.add_argument('--posebatch', type=int, default=80,
                       help='Pose estimation batch size')
    parser.add_argument('--vis_fast', action='store_true',
                       help='Use fast visualization')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--sp', action='store_true',
                       help='Use single process for pytorch')
    parser.add_argument('--extensions', nargs='+', 
                       default=['*.mp4', '*.avi', '*.mov', '*.mkv'],
                       help='Video file extensions to process')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already processed videos')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        return
    
    # Get all video files
    print(f"Searching for videos in: {args.input_dir}")
    video_files = get_video_files(args.input_dir, args.extensions)
    
    if not video_files:
        print(f"ERROR: No video files found in {args.input_dir}")
        print(f"Searched for extensions: {args.extensions}")
        return
    
    print(f"Found {len(video_files)} video files:")
    for i, video in enumerate(video_files, 1):
        print(f"  {i:2d}. {os.path.relpath(video, args.input_dir)}")
    
    # Build additional arguments
    additional_args = [
        '--mode', args.mode,
        '--conf', str(args.conf),
        '--nms', str(args.nms),
        '--detbatch', str(args.detbatch),
        '--posebatch', str(args.posebatch)
    ]
    
    if args.vis_fast:
        additional_args.append('--vis_fast')
    if args.cpu:
        additional_args.append('--cpu')
    if args.sp:
        additional_args.append('--sp')
    
    # Process videos
    successful = 0
    failed = 0
    skipped = 0
    
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[VIDEO {i}/{len(video_files)}] Processing...")
        
        # Check if already processed (resume functionality)
        if args.resume:
            video_name = Path(video_path).stem
            expected_output = os.path.join(args.output_dir, f'AlphaPose_{video_name}.avi')
            expected_json = os.path.join(args.output_dir, f'{video_name}.json')
            
            if os.path.exists(expected_output) and os.path.exists(expected_json):
                print(f"SKIPPING: {video_name} (already processed)")
                skipped += 1
                continue
        
        # Process the video
        if process_single_video(video_path, args.output_dir, additional_args):
            successful += 1
        else:
            failed += 1
            print(f"ERROR: Failed to process {os.path.basename(video_path)}")
    
    total_end_time = time.time()
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos found: {len(video_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
