def gen_plots(history, model, dataset):

    # 1. Training Progress Graph
    fig1 = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='MSE Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['pck'], label='PCK@0.1', color='orange')
    plt.title('Pose Accuracy (PCK)')
    plt.legend()
    plt.tight_layout() # Ensure proper layout
    plt.savefig('training_results.png')
    plt.show()
    plt.close(fig1) # Close the figure to release resources

    # 2. Skeleton Prediction Visualization
    model.eval()
    radar_sample, gt_keypoints_sample = dataset[0]
    with torch.no_grad():
        # Prepare radar input for the model: add batch and channel dims, move to device
        radar_input = radar_sample.unsqueeze(0).unsqueeze(0).to(device)
        pred = model(radar_input).view(-1)[:34].cpu().numpy() # Get predictions and move to CPU for numpy

    gt = gt_keypoints_sample[:34].numpy()

    fig2 = plt.figure(figsize=(6, 6))
    plt.scatter(gt[0::2], gt[1::2], c='g', label='Ground Truth')
    plt.scatter(pred[0::2], pred[1::2], c='r', marker='x', label='Predicted')
    plt.title("Keypoint Localization Result")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout() # Ensure proper layout
    plt.savefig('localization_result.png')
    plt.show()
    plt.close(fig2) # Close the figure to release resources

# Run full training and evaluation
history = train_and_evaluate(model, loader, epochs=10)
gen_plots(history, model, dataset)
