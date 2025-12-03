"""
Callbacks for controlling training processes.

Example:
    >>> early_stopping = EarlyStopping(patience=5, delta=0.01)
    >>> for epoch in range(epochs):
    ...     val_loss = validate()
    ...     if early_stopping(val_loss):
    ...         break
"""


class EarlyStopping:
    """
    Stops training when validation loss plateaus or increases.

    Monitors validation loss and triggers early stopping after 'patience' epochs
    without improvement of at least 'delta'.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.01,
    ) -> None:
        """
        Args:
            patience: Number of epochs without improvement before stopping.
            delta: Minimum loss reduction to count as improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if early stopping should trigger, False otherwise.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True
        return self.early_stop
