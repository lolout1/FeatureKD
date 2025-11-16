"""
Comprehensive Model Comparison Script with LOOCV Support
Compares Accelerometer-only vs Full IMU (Acc+Gyro+SMV) Transformers

Supports:
- Leave-One-Subject-Out Cross-Validation (LOOCV)
- Comprehensive per-fold logging and metrics
- Modular, scalable architecture
- Detailed reporting with enhanced metrics

Usage:
    # LOOCV mode
    python compare_models.py --baseline config/smartfallmm/comparison_acc_only.yaml --imu config/smartfallmm/comparison_imu_full.yaml --loocv

    # Single split mode
    python compare_models.py --baseline config/smartfallmm/comparison_acc_only.yaml --imu config/smartfallmm/comparison_imu_full.yaml
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
import argparse
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Import enhanced reporting utilities
from utils.metrics_report import (
    save_enhanced_results, generate_text_report,
    format_per_fold_table, merge_model_results
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class LOOCVModelComparison:
    """
    Handles LOOCV training, evaluation, and comparison of multiple models.

    This class implements Leave-One-Subject-Out cross-validation for fair
    model comparison with comprehensive logging and reporting.
    """

    def __init__(self, output_dir: str = None, loocv: bool = True):
        """
        Initialize comparison framework

        Args:
            output_dir: Directory to save results and reports
            loocv: Whether to use LOOCV or single split
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"work_dir/model_comparison_{timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loocv = loocv

        # Create subdirectories
        self.models_dir = self.output_dir / "models"
        self.plots_dir = self.output_dir / "plots"
        self.configs_dir = self.output_dir / "configs"
        self.reports_dir = self.output_dir / "reports"

        for dir_path in [self.models_dir, self.plots_dir, self.configs_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)

        # Logging
        self.log_file = self.output_dir / "comparison.log"
        self.log_handle = open(self.log_file, 'w')

        # Results storage
        self.results = {}

    def print_log(self, msg: str, print_to_console: bool = True):
        """Print message to both log file and console"""
        if print_to_console:
            print(msg)
        self.log_handle.write(msg + '\n')
        self.log_handle.flush()

    def load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Save copy of config
        config_name = Path(config_path).name
        with open(self.configs_dir / config_name, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config

    def create_model(self, config: Dict) -> torch.nn.Module:
        """
        Create model from configuration

        Args:
            config: Configuration dictionary

        Returns:
            Initialized model
        """
        model_path = config['model']
        module_name, class_name = model_path.rsplit('.', 1)

        # Import module and get class
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # Create model with args
        model_args = config.get('model_args', {})
        model = model_class(**model_args)

        return model

    def create_dataloader(
        self,
        config: Dict,
        split: str = 'train',
        subjects: List[int] = None
    ) -> Optional[torch.utils.data.DataLoader]:
        """
        Create dataloader from configuration with optional subject filtering

        Args:
            config: Configuration dictionary
            split: 'train', 'val', or 'test'
            subjects: List of subject IDs to include (overrides config)

        Returns:
            DataLoader or None if creation fails
        """
        feeder_path = config['feeder']
        module_name, class_name = feeder_path.rsplit('.', 1)

        # Import feeder
        module = importlib.import_module(module_name)
        feeder_class = getattr(module, class_name)

        # Get appropriate args
        feeder_args = config.get(f'{split}_feeder_args', {})
        dataset_args = config.get('dataset_args', {})

        # Merge args
        merged_args = {**dataset_args, **feeder_args}
        merged_args['split'] = split
        merged_args['dataset'] = config.get('dataset', 'smartfallmm')

        # Override subjects if provided
        if subjects is not None:
            merged_args['subjects'] = subjects
        elif 'subjects' in config:
            merged_args['subjects'] = config['subjects']

        # Handle validation subjects
        if 'validation_subjects' in config and split == 'val' and subjects is None:
            merged_args['subjects'] = config['validation_subjects']

        # Create dataset and dataloader
        try:
            dataset = feeder_class(**merged_args)

            batch_size = config.get(f'{split}_batch_size',
                                   config.get('batch_size', 32))

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=4,
                pin_memory=True
            )

            return dataloader

        except Exception as e:
            self.print_log(f"Warning: Could not create {split} dataloader: {e}")
            return None

    def train_single_fold(
        self,
        model: torch.nn.Module,
        config: Dict,
        model_name: str,
        fold_idx: int,
        test_subject: int,
        train_subjects: List[int],
        val_subjects: List[int],
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Train a model for a single LOOCV fold

        Args:
            model: Model to train
            config: Configuration dictionary
            model_name: Name for saving
            fold_idx: Fold index (0-based)
            test_subject: Subject ID for testing
            train_subjects: List of subject IDs for training
            val_subjects: List of subject IDs for validation
            device: Device to use

        Returns:
            Dictionary with fold results
        """
        self.print_log(f"\n{'='*80}")
        self.print_log(f"Fold {fold_idx + 1} - {model_name}")
        self.print_log(f"Test Subject: {test_subject}")
        self.print_log(f"Train Subjects: {train_subjects}")
        self.print_log(f"Val Subjects: {val_subjects}")
        self.print_log(f"{'='*80}")

        model = model.to(device)

        # Create dataloaders with subject filtering
        train_loader = self.create_dataloader(config, 'train', subjects=train_subjects)
        val_loader = self.create_dataloader(config, 'val', subjects=val_subjects)
        test_loader = self.create_dataloader(config, 'test', subjects=[test_subject])

        if train_loader is None or val_loader is None or test_loader is None:
            self.print_log(f"Error: Could not create dataloaders for fold {fold_idx}")
            return {}

        # Setup optimizer
        num_epochs = config.get('num_epoch', 50)
        lr = config.get('base_lr', 1e-3)
        weight_decay = config.get('weight_decay', 1e-3)
        optimizer_name = config.get('optimizer', 'adamw').lower()

        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Setup loss function
        num_classes = config['model_args'].get('num_classes', 2)
        if num_classes == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        # Training history
        epoch_logs = []
        best_val_f1 = 0.0
        best_epoch = 0
        best_model_state = None
        best_val_metrics = None

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(
                model, train_loader, criterion, optimizer, device, num_classes, epoch, num_epochs
            )

            # Validation phase
            val_metrics = self._evaluate_epoch(
                model, val_loader, criterion, device, num_classes, "Val"
            )

            # Log epoch metrics
            epoch_log = {
                'fold': fold_idx + 1,
                'test_subject': test_subject,
                'epoch': epoch + 1,
                'phase': 'train',
                **train_metrics
            }
            epoch_logs.append(epoch_log)

            epoch_log = {
                'fold': fold_idx + 1,
                'test_subject': test_subject,
                'epoch': epoch + 1,
                'phase': 'val',
                **val_metrics
            }
            epoch_logs.append(epoch_log)

            # Print epoch summary
            self.print_log(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%, F1: {train_metrics['f1_score']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, F1: {val_metrics['f1_score']:.2f}%"
            )

            # Save best model based on validation F1
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                best_epoch = epoch + 1
                best_model_state = deepcopy(model.state_dict())
                best_val_metrics = deepcopy(val_metrics)

            # Step scheduler
            scheduler.step()

        self.print_log(f"Best validation F1: {best_val_f1:.2f}% at epoch {best_epoch}")

        # Load best model for testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Test phase
        test_metrics = self._evaluate_epoch(
            model, test_loader, criterion, device, num_classes, "Test"
        )

        self.print_log(f"Test Results - Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1_score']:.2f}%")

        # Save fold model
        fold_model_path = self.models_dir / f"{model_name}_fold{fold_idx+1}_subject{test_subject}.pth"
        torch.save({
            'fold': fold_idx,
            'test_subject': test_subject,
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'train_metrics': train_metrics,
            'val_metrics': best_val_metrics,
            'test_metrics': test_metrics,
            'config': config
        }, fold_model_path)

        # Return fold results
        fold_results = {
            'fold': fold_idx,
            'test_subject': test_subject,
            'train': train_metrics,
            'val': best_val_metrics,
            'test': test_metrics,
            'best_epoch': best_epoch,
            'epoch_logs': epoch_logs
        }

        return fold_results

    def _train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        num_classes: int,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """Train for one epoch and return metrics"""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
        for batch in pbar:
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    acc_data, skl_data, labels = batch
                elif len(batch) == 2:
                    acc_data, labels = batch
                    skl_data = None
                else:
                    continue
            else:
                continue

            acc_data = acc_data.to(device)
            labels = labels.to(device)
            if skl_data is not None:
                skl_data = skl_data.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs, features = model(acc_data, skl_data, epoch=epoch)

            # Compute loss
            if num_classes == 1:
                loss = criterion(outputs.squeeze(), labels.float())
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs > 0.5).long()
            else:
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if num_classes == 1:
                all_probs.extend(probs.cpu().numpy())
            else:
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * accuracy_score(all_labels, all_preds)
        precision = 100. * precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = 100. * recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = 100. * f1_score(all_labels, all_preds, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

    def _evaluate_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: str,
        num_classes: int,
        phase_name: str = "Eval"
    ) -> Dict[str, float]:
        """Evaluate model and return metrics"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        acc_data, skl_data, labels = batch
                    elif len(batch) == 2:
                        acc_data, labels = batch
                        skl_data = None
                    else:
                        continue
                else:
                    continue

                acc_data = acc_data.to(device)
                labels = labels.to(device)
                if skl_data is not None:
                    skl_data = skl_data.to(device)

                # Forward pass
                outputs, _ = model(acc_data, skl_data)

                # Compute loss
                if num_classes == 1:
                    loss = criterion(outputs.squeeze(), labels.float())
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs > 0.5).long()
                else:
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)

                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if num_classes == 1:
                    all_probs.extend(probs.cpu().numpy())
                else:
                    all_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * accuracy_score(all_labels, all_preds)
        precision = 100. * precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = 100. * recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = 100. * f1_score(all_labels, all_preds, average='binary', zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

    def run_loocv(
        self,
        config: Dict,
        model_name: str,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Run Leave-One-Subject-Out cross-validation

        Args:
            config: Model configuration
            model_name: Name of the model
            device: Device to use

        Returns:
            Dictionary with LOOCV results
        """
        self.print_log(f"\n{'='*100}")
        self.print_log(f"Starting LOOCV for {model_name}")
        self.print_log(f"{'='*100}\n")

        # Get subjects
        all_subjects = config.get('subjects', [])
        val_subjects = config.get('validation_subjects', [])

        # Filter out validation subjects from LOOCV
        available_subjects = [s for s in all_subjects if s not in val_subjects]

        self.print_log(f"Total subjects: {len(all_subjects)}")
        self.print_log(f"Validation subjects (held out): {val_subjects}")
        self.print_log(f"Available subjects for LOOCV: {available_subjects}")
        self.print_log(f"Number of folds: {len(available_subjects)}\n")

        # Storage for all fold results
        fold_results_list = []
        all_epoch_logs = []

        # Run LOOCV
        for fold_idx, test_subject in enumerate(available_subjects):
            # Create fresh model for this fold
            model = self.create_model(config)

            # Get train subjects (all except test and validation subjects)
            train_subjects = [s for s in available_subjects if s != test_subject]

            # Train and evaluate fold
            fold_results = self.train_single_fold(
                model=model,
                config=config,
                model_name=model_name,
                fold_idx=fold_idx,
                test_subject=test_subject,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                device=device
            )

            if fold_results:
                fold_results_list.append(fold_results)
                all_epoch_logs.extend(fold_results['epoch_logs'])

        # Aggregate results
        loocv_results = self._aggregate_loocv_results(
            fold_results_list, all_epoch_logs, model_name
        )

        return loocv_results

    def _aggregate_loocv_results(
        self,
        fold_results: List[Dict],
        epoch_logs: List[Dict],
        model_name: str
    ) -> Dict[str, Any]:
        """Aggregate LOOCV results across all folds"""

        # Convert to format expected by metrics_report
        fold_metrics = []
        for fold in fold_results:
            fold_metrics.append({
                'test_subject': str(fold['test_subject']),
                'train': fold['train'],
                'val': fold['val'],
                'test': fold['test']
            })

        # Create per-fold DataFrame
        per_fold_df = format_per_fold_table(fold_metrics)

        # Calculate aggregated statistics
        test_metrics = {
            'accuracy_mean': per_fold_df['test_accuracy'].mean(),
            'accuracy_std': per_fold_df['test_accuracy'].std(),
            'f1_mean': per_fold_df['test_f1_score'].mean(),
            'f1_std': per_fold_df['test_f1_score'].std(),
            'precision_mean': per_fold_df['test_precision'].mean(),
            'precision_std': per_fold_df['test_precision'].std(),
            'recall_mean': per_fold_df['test_recall'].mean(),
            'recall_std': per_fold_df['test_recall'].std(),
        }

        # Save results
        scores_csv_path = self.output_dir / f"{model_name}_scores.csv"
        per_fold_df.to_csv(scores_csv_path, index=False, float_format='%.2f')
        self.print_log(f"\nSaved per-fold scores to: {scores_csv_path}")

        # Save epoch logs
        epoch_logs_df = pd.DataFrame(epoch_logs)
        epoch_log_path = self.output_dir / f"{model_name}_training_log.csv"
        epoch_logs_df.to_csv(epoch_log_path, index=False, float_format='%.4f')
        self.print_log(f"Saved training logs to: {epoch_log_path}")

        # Generate enhanced reports using metrics_report utilities
        model_report_dir = self.reports_dir / model_name
        model_report_dir.mkdir(exist_ok=True)

        save_enhanced_results(
            fold_metrics=fold_metrics,
            output_dir=str(model_report_dir),
            model_name=model_name
        )

        # Print summary
        summary_text = generate_text_report(fold_metrics, model_name)
        self.print_log("\n" + summary_text)

        return {
            'fold_results': fold_results,
            'fold_metrics': fold_metrics,
            'per_fold_df': per_fold_df,
            'aggregated_metrics': test_metrics,
            'epoch_logs': epoch_logs
        }

    def run_single_split(
        self,
        config: Dict,
        model_name: str,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Run single train/val/test split (non-LOOCV mode)

        Args:
            config: Model configuration
            model_name: Name of the model
            device: Device to use

        Returns:
            Dictionary with results
        """
        self.print_log(f"\n{'='*100}")
        self.print_log(f"Training {model_name} (Single Split Mode)")
        self.print_log(f"{'='*100}\n")

        model = self.create_model(config)
        model = model.to(device)

        # Create dataloaders (use config defaults)
        train_loader = self.create_dataloader(config, 'train')
        val_loader = self.create_dataloader(config, 'val')
        test_loader = self.create_dataloader(config, 'test')

        if train_loader is None:
            self.print_log(f"Error: Could not create dataloaders for {model_name}")
            return {}

        # Setup optimizer
        num_epochs = config.get('num_epoch', 50)
        lr = config.get('base_lr', 1e-3)
        weight_decay = config.get('weight_decay', 1e-3)
        optimizer_name = config.get('optimizer', 'adamw').lower()

        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Setup loss function
        num_classes = config['model_args'].get('num_classes', 2)
        if num_classes == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Training
        best_val_f1 = 0.0
        best_epoch = 0
        best_model_state = None

        for epoch in range(num_epochs):
            train_metrics = self._train_epoch(
                model, train_loader, criterion, optimizer, device, num_classes, epoch, num_epochs
            )

            if val_loader:
                val_metrics = self._evaluate_epoch(
                    model, val_loader, criterion, device, num_classes, "Val"
                )
            else:
                val_metrics = train_metrics

            self.print_log(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%"
            )

            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                best_epoch = epoch + 1
                best_model_state = deepcopy(model.state_dict())

            scheduler.step()

        # Test
        if best_model_state:
            model.load_state_dict(best_model_state)

        test_metrics = self._evaluate_epoch(
            model, test_loader, criterion, device, num_classes, "Test"
        ) if test_loader else {}

        self.print_log(f"\nBest Val F1: {best_val_f1:.2f}% at epoch {best_epoch}")
        self.print_log(f"Test Acc: {test_metrics.get('accuracy', 0):.2f}%, F1: {test_metrics.get('f1_score', 0):.2f}%")

        # Save model
        torch.save({
            'model_state_dict': best_model_state,
            'test_metrics': test_metrics,
            'config': config
        }, self.models_dir / f"{model_name}_best.pth")

        return {
            'test_metrics': test_metrics,
            'best_epoch': best_epoch
        }

    def compare_models(
        self,
        configs: Dict[str, Dict],
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Compare multiple models using LOOCV or single split

        Args:
            configs: Dictionary mapping model_name -> config
            device: Device to use

        Returns:
            Comparison results
        """
        all_results = {}

        for model_name, config in configs.items():
            self.print_log(f"\n{'#'*100}")
            self.print_log(f"# Processing Model: {model_name}")
            self.print_log(f"{'#'*100}")

            # Count parameters
            model = self.create_model(config)
            num_params = sum(p.numel() for p in model.parameters())
            self.print_log(f"Model Parameters: {num_params:,}")
            del model

            # Run LOOCV or single split
            if self.loocv:
                results = self.run_loocv(config, model_name, device)
            else:
                results = self.run_single_split(config, model_name, device)

            all_results[model_name] = results

        # Generate comparison report
        if self.loocv:
            self._generate_loocv_comparison_report(all_results, configs)
        else:
            self._generate_single_split_comparison_report(all_results, configs)

        return all_results

    def _generate_loocv_comparison_report(
        self,
        all_results: Dict[str, Any],
        configs: Dict[str, Dict]
    ):
        """Generate comparison report for LOOCV results"""
        self.print_log(f"\n{'='*100}")
        self.print_log("FINAL LOOCV COMPARISON SUMMARY")
        self.print_log(f"{'='*100}\n")

        # Prepare comparison data
        model_dfs = {}
        for model_name, results in all_results.items():
            if 'per_fold_df' in results:
                model_dfs[model_name] = results['per_fold_df']

        # Create merged comparison table
        if len(model_dfs) > 0:
            merged_df = merge_model_results(model_dfs)
            merged_csv_path = self.reports_dir / "cross_model_comparison.csv"
            merged_df.to_csv(merged_csv_path, index=False, float_format='%.2f')
            self.print_log(f"Saved cross-model comparison to: {merged_csv_path}\n")

        # Print summary table
        summary_data = []
        for model_name, results in all_results.items():
            if 'aggregated_metrics' in results:
                metrics = results['aggregated_metrics']
                summary_data.append({
                    'Model': model_name,
                    'Test Accuracy': f"{metrics['accuracy_mean']:.2f} ± {metrics['accuracy_std']:.2f}",
                    'Test F1': f"{metrics['f1_mean']:.2f} ± {metrics['f1_std']:.2f}",
                    'Test Precision': f"{metrics['precision_mean']:.2f} ± {metrics['precision_std']:.2f}",
                    'Test Recall': f"{metrics['recall_mean']:.2f} ± {metrics['recall_std']:.2f}"
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            self.print_log("\nModel Comparison Summary:")
            self.print_log(summary_df.to_string(index=False))

            summary_csv_path = self.reports_dir / "summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            self.print_log(f"\nSaved summary to: {summary_csv_path}")

        self.print_log(f"\n{'='*100}")
        self.print_log("Comparison Complete!")
        self.print_log(f"All results saved to: {self.output_dir}")
        self.print_log(f"{'='*100}\n")

    def _generate_single_split_comparison_report(
        self,
        all_results: Dict[str, Any],
        configs: Dict[str, Dict]
    ):
        """Generate comparison report for single split results"""
        self.print_log(f"\n{'='*100}")
        self.print_log("FINAL COMPARISON SUMMARY (Single Split)")
        self.print_log(f"{'='*100}\n")

        summary_data = []
        for model_name, results in all_results.items():
            if 'test_metrics' in results:
                metrics = results['test_metrics']
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics.get('accuracy', 0):.2f}%",
                    'F1': f"{metrics.get('f1_score', 0):.2f}%",
                    'Precision': f"{metrics.get('precision', 0):.2f}%",
                    'Recall': f"{metrics.get('recall', 0):.2f}%"
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            self.print_log("\nModel Comparison:")
            self.print_log(summary_df.to_string(index=False))

            summary_csv_path = self.reports_dir / "summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)

        self.print_log(f"\n{'='*100}")
        self.print_log("Comparison Complete!")
        self.print_log(f"All results saved to: {self.output_dir}")
        self.print_log(f"{'='*100}\n")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'log_handle') and self.log_handle:
            self.log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Compare transformer models with LOOCV support")
    parser.add_argument(
        '--configs',
        nargs='+',
        help='List of config files to compare'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        help='Baseline config (accelerometer-only)'
    )
    parser.add_argument(
        '--imu',
        type=str,
        help='IMU config (acc+gyro+smv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--loocv',
        action='store_true',
        help='Use Leave-One-Subject-Out cross-validation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine configs
    if args.baseline and args.imu:
        config_files = [args.baseline, args.imu]
        model_names = ['ACC_Only', 'IMU_Full']
    elif args.configs:
        config_files = args.configs
        model_names = [f"Model_{i+1}" for i in range(len(config_files))]
    else:
        # Default configs
        config_files = [
            'config/smartfallmm/comparison_acc_only.yaml',
            'config/smartfallmm/comparison_imu_full.yaml'
        ]
        model_names = ['ACC_Only', 'IMU_Full']

    # Initialize comparison
    comparison = LOOCVModelComparison(output_dir=args.output_dir, loocv=args.loocv)

    comparison.print_log(f"{'='*100}")
    comparison.print_log(f"MODEL COMPARISON - {'LOOCV' if args.loocv else 'Single Split'} Mode")
    comparison.print_log(f"{'='*100}")
    comparison.print_log(f"Comparing {len(config_files)} models:")
    for name, config_file in zip(model_names, config_files):
        comparison.print_log(f"  - {name}: {config_file}")
    comparison.print_log(f"Random seed: {args.seed}")
    comparison.print_log(f"Device: {args.device}")
    comparison.print_log(f"{'='*100}\n")

    # Load configs
    configs = {}
    for name, config_file in zip(model_names, config_files):
        configs[name] = comparison.load_config(config_file)

    # Run comparison
    results = comparison.compare_models(configs, device=args.device)

    comparison.print_log("\n✓ All comparisons completed successfully!")


if __name__ == "__main__":
    main()
