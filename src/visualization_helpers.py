"""
Helper functions for visualizing neural network predictions and structure.
These functions can be used to display MNIST digits and network architecture.
"""
import numpy as np
# Set matplotlib to use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a display
import matplotlib.pyplot as plt
import random
import os
import shutil

def display_digit(image_data, index, predicted=None, actual=None, save_dir="output", display=True):
    """Display a single digit image with labels for predicted and actual values"""
    # Reshape the 784x1 vector to 28x28 image
    image = image_data.reshape(28, 28)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    
    title = ""
    if predicted is not None:
        title += f"Predicted: {predicted}"
    if actual is not None:
        title += f" | Actual: {actual}"
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Save to file
    filename = f"{save_dir}/digit_{index}_pred_{predicted}_actual_{actual}.png"
    plt.savefig(filename)
    
    # In a web server environment, we don't want to display windows
    # Always close the figure to free memory and prevent GUI issues
    plt.close()  
    
    if display:
        print(f"Image saved to {filename}")
    else:
        print(f"Image saved to {filename}")

def run_specific_examples(net, test_data, num_examples=5, specific_indices=None, display=True):
    """
    Run specific examples through the network and display results.
    
    Parameters:
    - net: The trained neural network
    - test_data: List of (x, y) test examples
    - num_examples: Number of random examples to display if specific_indices is None
    - specific_indices: List of specific indices from test_data to display
    - display: Whether to display the images or just save them
    """
    if specific_indices is None:
        # Choose random examples if no specific indices given
        specific_indices = random.sample(range(len(test_data)), num_examples)
    
    for idx in specific_indices:
        x, y = test_data[idx]
        
        # Get the network's output (a 10x1 vector of activations)
        output = net.feedforward(x)
        
        # The predicted digit is the index with highest activation
        predicted_digit = np.argmax(output)
        
        # The actual digit
        actual_digit = y
        
        print(f"Example {idx}:")
        print(f"  Network output: {[round(float(val), 4) for val in output]}")
        print(f"  Predicted digit: {predicted_digit}")
        print(f"  Actual digit: {actual_digit}")
        
        # Display the digit image with predictions
        display_digit(x, idx, predicted_digit, actual_digit, display=display)
        print()  # Empty line after each example for better readability

def visualize_network_structure(net, save_dir="output", display=True):
    """
    Visualize the entire network structure with nodes as dots and connections as lines,
    where line thickness represents weight magnitude.
    
    Parameters:
    - net: The trained neural network
    - save_dir: Directory to save the visualization
    - display: Whether to display the visualization or just save it
    """
    from matplotlib.collections import LineCollection
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get the sizes of each layer
    layer_sizes = [s for s in net.sizes]
    
    # Create a figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Define the positions of the nodes
    layer_positions = {}
    max_neurons = max(layer_sizes)
    
    # Create node positions for each layer
    for l, layer_size in enumerate(layer_sizes):
        # Horizontal position is based on layer index
        x = l
        
        # Calculate vertical positions to center each layer
        layer_positions[l] = []
        for n in range(layer_size):
            # Use a sigmoid-like spacing for layers with many neurons
            if layer_size > 50:
                if n < 10:  # Show the first few neurons
                    y = n / 10 * max_neurons
                elif n > layer_size - 10:  # Show the last few neurons
                    y = (n - (layer_size - 10)) / 10 * max_neurons + max_neurons * 0.9
                else:
                    # Indicate that neurons are skipped
                    if n == 10:
                        layer_positions[l].append((x, max_neurons * 0.5))
                    continue
            else:
                # For smaller layers, space evenly
                y = n * (max_neurons / (layer_size - 1 if layer_size > 1 else 1))
            
            layer_positions[l].append((x, y))
    
    # Draw connections between layers
    for l in range(len(layer_sizes) - 1):
        # Get the weight matrix for this layer
        weights = net.weights[l]
        abs_weights = np.abs(weights)
        
        # Normalize weights for line thickness
        max_weight = np.max(abs_weights)
        min_weight = np.min(abs_weights)
        norm_weights = (abs_weights - min_weight) / (max_weight - min_weight) if max_weight > min_weight else abs_weights
        
        # For layers with many neurons, we need to adjust the connection visualization
        source_positions = layer_positions[l]
        target_positions = layer_positions[l+1]
        
        # Draw connections
        lines = []
        line_colors = []
        line_widths = []
        
        for i, source_pos in enumerate(source_positions):
            for j, target_pos in enumerate(target_positions):
                # Skip if we're at an ellipsis position
                if source_pos[1] == max_neurons * 0.5 and i == 10:
                    continue
                if target_pos[1] == max_neurons * 0.5 and j == 10:
                    continue
                
                # Map indices for the weight matrix
                # For layers with many neurons, we've already filtered positions in the layout phase
                source_idx = i
                target_idx = j
                
                # Skip ellipsis points or invalid indices
                if source_idx >= weights.shape[0] or target_idx >= weights.shape[1]:
                    continue                # Valid indices, proceed
                weight = weights[source_idx, target_idx]
                norm_weight = norm_weights[source_idx, target_idx]
                
                # Add line with appropriate thickness
                lines.append([source_pos, target_pos])
                
                # Color based on sign of weight (red for negative, blue for positive)
                color = 'red' if weight < 0 else 'blue'
                line_colors.append(color)
                
                # Scale line width based on normalized weight (min 0.1, max 3.0)
                line_widths.append(0.1 + norm_weight * 2.9)
        
        # Create line collection for efficient drawing
        lc = LineCollection(lines, colors=line_colors, linewidths=line_widths, alpha=0.3)
        ax.add_collection(lc)
    
    # Draw nodes
    for l, layer_size in enumerate(layer_sizes):
        positions = layer_positions[l]
        
        # Draw the nodes
        x_vals = [pos[0] for pos in positions]
        y_vals = [pos[1] for pos in positions]
        
        # Determine node size based on number of layers
        node_size = max(80 / len(layer_sizes), 15)
        
        ax.scatter(x_vals, y_vals, s=node_size, c='lightgray', edgecolors='black', zorder=10)
        
        # Draw the ellipsis for skipped neurons if needed
        for pos in positions:
            if pos[1] == max_neurons * 0.5 and layer_size > 50:
                ax.text(pos[0], pos[1], "...", ha='center', va='center', fontsize=16)
        
        # Add layer labels
        if l == 0:
            ax.text(x_vals[0], -0.1 * max_neurons, "Input\nLayer", ha='center')
        elif l == len(layer_sizes) - 1:
            ax.text(x_vals[0], -0.1 * max_neurons, "Output\nLayer", ha='center')
        else:
            ax.text(x_vals[0], -0.1 * max_neurons, f"Hidden\nLayer {l}", ha='center')
    
    # Add legends/explanations
    ax.text(0.5, 1.05, "Network Architecture Visualization", 
            ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Add weight explanation
    ax.text(0.5, -0.05 * max_neurons, 
            "Line thickness represents weight magnitude | Blue: positive weights | Red: negative weights",
            ha='center', fontsize=10)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{save_dir}/network_structure.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # In a web server environment, we don't want to display windows
    # Always close the figure to free memory and prevent GUI issues
    plt.close()
    
    if display:
        print(f"Network structure visualization saved to {filename}")
    else:
        print(f"Network structure visualization saved to {filename}")

def find_misclassified_examples(net, test_data, max_count=3, max_check=200):
    """
    Find examples that the network misclassifies.
    
    Parameters:
    - net: The trained neural network
    - test_data: List of (x, y) test examples
    - max_count: Maximum number of misclassified examples to find
    - max_check: Maximum number of examples to check
    
    Returns:
    - List of indices of misclassified examples
    """
    misclassified = []
    for i in range(min(max_check, len(test_data))):
        x, y = test_data[i]
        output = net.feedforward(x)
        predicted = np.argmax(output)
        if predicted != y:
            misclassified.append(i)
            if len(misclassified) >= max_count:
                break
    
    return misclassified

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
