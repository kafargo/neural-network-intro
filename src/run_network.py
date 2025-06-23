"""
Visualize how the network classifies specific examples from the MNIST dataset.
This script allows you to see both the input digit image and the network's prediction.
"""
import mnist_loader
import network
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from visualization_helpers import (
    visualize_network_structure, 
    run_specific_examples, 
    find_misclassified_examples
)

def get_user_display_preference():
    """Ask the user if they want to display visualizations during execution."""
    while True:
        response = input("Display visualizations during execution? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")

def clear_output_directory(output_dir="output"):
    """Clear the output directory to start fresh."""
    if os.path.exists(output_dir):
        print(f"Clearing existing output directory: {output_dir}")
        # Remove all files in the directory
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    else:
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"Output directory is ready: {output_dir}")

def main():
    # Clear the output directory
    clear_output_directory()
    
    # Ask user for display preference
    display_during_execution = get_user_display_preference()
    print(f"Visualizations will {'be displayed' if display_during_execution else 'only be saved to the output directory'} during execution.")
    
    # Load the MNIST data
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("Data loaded successfully!")

    # Option 1: Train a new network
    print("Creating and training neural network...")
    net = network.Network([784, 128, 64, 10])
    # For quick testing, use just 5 epochs. Increase for better accuracy.
    net.SGD(training_data, 5, 10, 2.5, test_data=test_data)
    
    print("\nVisualizing network structure...")
    visualize_network_structure(net, display=display_during_execution)

    print("\nTesting specific examples:")
    
    # Test with 5 random examples
    run_specific_examples(net, test_data, num_examples=3, display=display_during_execution)
    
    # Find some misclassified digits
    print("\nFinding misclassified examples:")
    misclassified = find_misclassified_examples(net, test_data, max_count=3, max_check=200)
    
    if misclassified:
        print(f"Found {len(misclassified)} misclassified examples")
        run_specific_examples(net, test_data, specific_indices=misclassified, display=display_during_execution)
    else:
        print("No misclassified examples found in the first 200 test cases!")
    
    # Keep the windows open until user closes them if we're displaying
    if display_during_execution:
        print("\nAll images have been displayed. Close the image windows or press Ctrl+C to exit.")
        try:
            plt.show()
        except KeyboardInterrupt:
            print("Exiting...")
    else:
        print("\nAll images have been saved to the output directory.")

if __name__ == "__main__":
    main()
