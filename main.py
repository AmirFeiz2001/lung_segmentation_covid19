import argparse
from src.convert_nii_to_png import convert_nii_to_png
from src.data_processing import prepare_data, split_data
from src.model import AttentionUNet, compile_model
from src.train import train_model, evaluate_model
from src.visualize import plot_predictions

def main():
    parser = argparse.ArgumentParser(description="Lung Segmentation for COVID-19 CT Scans")
    parser.add_argument('--data_dir', type=str, default='data', help='Base directory for input data')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    args = parser.parse_args()

    # Convert .nii to .png
    dirs = {
        'ct_scans': 'ct_scans_png',
        'lung_mask': 'lung_mask_png',
        'infection_mask': 'infection_mask_png',
        'lung_and_infection_mask': 'lung_and_infection_mask_png'
    }
    for input_dir, output_dir in dirs.items():
        convert_nii_to_png(
            os.path.join(args.data_dir, input_dir),
            os.path.join(args.output_dir, output_dir),
            rotate_ct=(input_dir == 'ct_scans')
        )

    # Prepare data
    CT, Mask = prepare_data(
        os.path.join(args.output_dir, 'ct_scans_png'),
        os.path.join(args.output_dir, 'lung_mask_png')
    )
    CT_train, CT_test, Mask_train, Mask_test = split_data(CT, Mask)

    # Build and train model
    unet = AttentionUNet()
    model = compile_model(unet.build_unet())
    train_model(model, CT_train, Mask_train, epochs=args.epochs, output_dir=args.output_dir)

    # Evaluate
    evaluate_model(model, CT_test, Mask_test)

    # Predict and visualize
    predictions = model.predict(CT_test)
    plot_predictions(CT_test, Mask_test, predictions, output_dir=os.path.join(args.output_dir, 'plots'))

if __name__ == "__main__":
    main()
