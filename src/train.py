from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(model, X_train, y_train, epochs=50, batch_size=8, output_dir="output"):
    """Train the Attention U-Net model."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, "best_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[checkpoint, early, reduce_lr],
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    return loss, acc
