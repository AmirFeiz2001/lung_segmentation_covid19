import os
import matplotlib.pyplot as plt
from datetime import datetime

def plot_predictions(X_test, y_test, predictions, num_samples=5, output_dir="output/plots"):
    """Visualize original CT, ground truth, and predictions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(min(num_samples, len(X_test))):
        fig = plt.figure(figsize=(18, 15))
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i][..., 0], cmap='bone')
        plt.title('Original CT Image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(X_test[i][..., 0], cmap='bone')
        plt.imshow(y_test[i][..., 0], alpha=0.5, cmap='nipy_spectral')
        plt.title('Original Infection Mask')
        
        plt.subplot(1, 3, 3)
        plt.imshow(X_test[i][..., 0], cmap='bone')
        plt.imshow(predictions[i][..., 0], alpha=0.5, cmap='nipy_spectral')
        plt.title('Predicted Infection Mask')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f"prediction_{i}_{timestamp}.png"))
        plt.close()
    print(f"Plots saved to {output_dir}")
