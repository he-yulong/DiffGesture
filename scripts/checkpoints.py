# DiffGesture/scripts/checkpoints.py
from lightning.pytorch.callbacks import ModelCheckpoint

# Save top-3 models by validation loss
val_checkpoint = ModelCheckpoint(
    monitor="val/mse_loss",  # 👈 must match your logged validation loss key
    mode="min",  # smaller is better
    save_top_k=3,
    filename="val-loss-{epoch:03d}-{val/mse_loss:.4f}",
    save_last=False,
    auto_insert_metric_name=False
)

# Save top-3 models by training loss
train_checkpoint = ModelCheckpoint(
    monitor="train/mse_loss_epoch",  # 👈 use epoch-level averaged training loss
    mode="min",
    save_top_k=3,
    filename="train-loss-{epoch:03d}-{train/mse_loss_epoch:.4f}",
    save_last=False,
    auto_insert_metric_name=False
)

# Always save the latest model
last_checkpoint = ModelCheckpoint(
    save_top_k=1,
    every_n_epochs=1,  # overwrite every epoch
    filename="latest-{epoch:03d}",
    save_last=True
)
