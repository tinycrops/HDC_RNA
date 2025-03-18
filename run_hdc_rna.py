#!/usr/bin/env python
"""
Main script for running the HDC RNA 3D structure prediction.
"""

import argparse
import os
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HDC RNA 3D Structure Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run a quick demonstration with sample data
  python run_hdc_rna.py demo
  
  # Train a model on the full dataset
  python run_hdc_rna.py train --data_dir stanford-rna-3d-folding --output_dir models
  
  # Train with limited sequences for faster execution
  python run_hdc_rna.py train --data_dir stanford-rna-3d-folding --max_sequences 10
  
  # Predict structure for a specific RNA sequence
  python run_hdc_rna.py predict --model_path models/rna_hdc_model.pt --sequence GGGUGCUCAGUACGAGAGGAACCGCACCC --visualize
  
  # Predict structure for a sequence from the dataset
  python run_hdc_rna.py predict --model_path models/rna_hdc_model.pt --target_id R1107 --visualize
  
  # Test the HDC implementation
  python run_hdc_rna.py test
'''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run a quick demonstration')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, default='stanford-rna-3d-folding',
                        help='Directory containing RNA data files')
    train_parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    train_parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to use for training')
    train_parser.add_argument('--sample', action='store_true',
                        help='Use a small sample of data for quick testing')
    train_parser.add_argument('--hdc_dimensions', type=int, default=10000,
                        help='Dimensionality of hypervectors')
    train_parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict RNA 3D structure')
    predict_parser.add_argument('--data_dir', type=str, default='stanford-rna-3d-folding',
                        help='Directory containing RNA data files')
    predict_parser.add_argument('--model_path', type=str, default='models/rna_hdc_model.pt',
                        help='Path to trained model file')
    predict_parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    predict_parser.add_argument('--target_id', type=str, default=None,
                        help='Target ID for prediction (if None, use validation sequences)')
    predict_parser.add_argument('--sequence', type=str, default=None,
                        help='RNA sequence for prediction (if provided, overrides target_id)')
    predict_parser.add_argument('--hdc_dimensions', type=int, default=10000,
                        help='Dimensionality of hypervectors (must match trained model)')
    predict_parser.add_argument('--visualize', action='store_true',
                        help='Generate 3D visualization of predicted structure')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test HDC implementation')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Add the parent directory to Python path to import our modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    if args.command == 'demo':
        # Run demonstration
        from hdc_rna.run_sample import main as demo_main
        demo_main()
        
    elif args.command == 'train':
        # Import and run the training script
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hdc_rna'))
        from hdc_rna.train_model import main as train_main
        
        # Override sys.argv to pass the arguments to the training script
        sys.argv = [
            'train_model.py',
            f'--data_dir={args.data_dir}',
            f'--output_dir={args.output_dir}',
            f'--hdc_dimensions={args.hdc_dimensions}',
            f'--batch_size={args.batch_size}',
            f'--epochs={args.epochs}'
        ]
        
        if args.max_sequences:
            sys.argv.append(f'--max_sequences={args.max_sequences}')
            
        if args.sample:
            sys.argv.append('--sample')
            
        train_main()
        
    elif args.command == 'predict':
        # Import and run the prediction script
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hdc_rna'))
        from hdc_rna.predict import main as predict_main
        
        # Override sys.argv to pass the arguments to the prediction script
        sys.argv = [
            'predict.py',
            f'--data_dir={args.data_dir}',
            f'--model_path={args.model_path}',
            f'--output_dir={args.output_dir}',
            f'--hdc_dimensions={args.hdc_dimensions}'
        ]
        
        if args.target_id:
            sys.argv.append(f'--target_id={args.target_id}')
            
        if args.sequence:
            sys.argv.append(f'--sequence={args.sequence}')
            
        if args.visualize:
            sys.argv.append('--visualize')
            
        predict_main()
        
    elif args.command == 'test':
        # Run the test script
        from hdc_rna.test_hdc import main as test_main
        test_main()
        
    else:
        print("Please specify a command: demo, train, predict, or test")
        print("Use the -h flag for more information")
        
if __name__ == '__main__':
    main() 