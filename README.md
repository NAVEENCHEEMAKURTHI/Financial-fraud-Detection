import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
                           QMessageBox, QProgressBar, QSpinBox, QDoubleSpinBox, QFormLayout,
                           QGroupBox, QHeaderView, QTextEdit, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

class ModelTrainingThread(QThread):
    """Thread for training the Random Forest model without blocking the UI"""
    progress_update = pyqtSignal(int)
    training_complete = pyqtSignal(object, float, str)
    
    def _init_(self, X_train, X_test, y_train, y_test, n_estimators=100):
        super()._init_()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_estimators = n_estimators
        
    def run(self):
        # Training the model
        model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        
        # Simulate progress updates (in a real app, you'd check actual training progress)
        for i in range(10):
            self.progress_update.emit(i * 10)
            self.msleep(200)  # Simulate training time
        
        model.fit(self.X_train, self.y_train)
        
        # Final evaluation
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        self.progress_update.emit(100)
        self.training_complete.emit(model, accuracy, report)

class SimpleFraudDetectionApp(QMainWindow):
    def _init_(self):
        super()._init_()
        self.setWindowTitle("Healthcare Fraud Detection System")
        self.setMinimumSize(1000, 700)
        
        # Initialize instance variables
        self.data = None
        self.model = None
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importances = None
        self.scaler = StandardScaler()
        
        # Create sample data if needed
        self.create_sample_data()
        
        # Setup UI
        self.setup_ui()
    
    def create_sample_data(self):
        """Create sample dataset for demonstration"""
        try:
            if not os.path.exists('sample_healthcare_claims.csv'):
                # Number of records
                n = 1000
                
                # Create random data
                data = {
                    'claim_id': [f'CLM{i:06d}' for i in range(1, n+1)],
                    'provider_id': [f'PRV{np.random.randint(1, 100):03d}' for _ in range(n)],
                    'claim_amount': np.random.uniform(100, 10000, n).round(2),
                    'patient_age': np.random.randint(18, 90, n),
                    'procedure_code': [f'P{np.random.randint(100, 999)}' for _ in range(n)],
                    'days_in_hospital': np.random.randint(0, 30, n),
                    'previous_claims': np.random.randint(0, 20, n)
                }
                
                # Generate some features that might indicate fraud
                data['unusual_billing'] = np.random.randint(0, 2, n)
                data['multiple_procedures'] = np.random.randint(0, 3, n)
                data['out_of_network'] = np.random.randint(0, 2, n)
                
                # Create fraud label (for demonstration)
                # Higher probability of fraud when unusual_billing=1, multiple_procedures>1, and high claim_amount
                fraud_prob = (data['unusual_billing'] * 0.3 + 
                             (data['multiple_procedures'] > 1) * 0.4 + 
                             (data['claim_amount'] > 5000) * 0.3)
                data['is_fraud'] = (fraud_prob > 0.5).astype(int)
                
                # Create DataFrame
                df = pd.DataFrame(data)
                
                # Save to CSV
                df.to_csv('sample_healthcare_claims.csv', index=False)
                print("Sample data created and saved to 'sample_healthcare_claims.csv'")
        except Exception as e:
            print(f"Error creating sample data: {str(e)}")
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        self.data_tab = QWidget()
        self.training_tab = QWidget()
        self.detection_tab = QWidget()
        
        tabs.addTab(self.data_tab, "Data Management")
        tabs.addTab(self.training_tab, "Model Training")
        tabs.addTab(self.detection_tab, "Fraud Detection")
        
        # Set up individual tabs
        self.setup_data_tab()
        self.setup_training_tab()
        self.setup_detection_tab()
        
        main_layout.addWidget(tabs)
        self.setCentralWidget(main_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def setup_data_tab(self):
        layout = QVBoxLayout(self.data_tab)
        
        # Data import section
        import_group = QGroupBox("Data Import")
        import_layout = QHBoxLayout()
        
        self.file_path_label = QLabel("No file selected")
        import_btn = QPushButton("Import Data")
        import_btn.clicked.connect(self.import_data)
        
        # Button to load sample data
        sample_btn = QPushButton("Load Sample Data")
        sample_btn.clicked.connect(self.load_sample_data)
        
        import_layout.addWidget(self.file_path_label)
        import_layout.addWidget(import_btn)
        import_layout.addWidget(sample_btn)
        import_group.setLayout(import_layout)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        self.data_info_label = QLabel("No data loaded")
        
        preview_layout.addWidget(self.data_table)
        preview_layout.addWidget(self.data_info_label)
        preview_group.setLayout(preview_layout)
        
        layout.addWidget(import_group)
        layout.addWidget(preview_group)
    
    def setup_training_tab(self):
        layout = QVBoxLayout(self.training_tab)
        
        # Model configuration
        config_group = QGroupBox("Model Configuration")
        config_layout = QFormLayout()
        
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        self.n_estimators_spin.setSingleStep(10)
        
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setValue(0.3)
        self.test_size_spin.setSingleStep(0.05)
        
        config_layout.addRow("Number of Trees:", self.n_estimators_spin)
        config_layout.addRow("Test Size:", self.test_size_spin)
        config_group.setLayout(config_layout)
        
        # Training controls
        training_group = QGroupBox("Model Training")
        training_layout = QVBoxLayout()
        
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_model)
        
        self.train_progress = QProgressBar()
        
        training_layout.addWidget(train_btn)
        training_layout.addWidget(self.train_progress)
        training_group.setLayout(training_layout)
        
        # Results display
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        # Feature importance
        importance_group = QGroupBox("Feature Importance")
        importance_layout = QVBoxLayout()
        
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(2)
        self.importance_table.setHorizontalHeaderLabels(["Feature", "Importance"])
        self.importance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        importance_layout.addWidget(self.importance_table)
        importance_group.setLayout(importance_layout)
        
        # Add all to layout
        layout.addWidget(config_group)
        layout.addWidget(training_group)
        layout.addWidget(results_group)
        layout.addWidget(importance_group)
    
    def setup_detection_tab(self):
        layout = QVBoxLayout(self.detection_tab)
        
        # Sample claim testing
        sample_group = QGroupBox("Test Sample Claims")
        sample_layout = QVBoxLayout()
        
        detect_btn = QPushButton("Run Fraud Detection on Test Data")
        detect_btn.clicked.connect(self.detect_fraud_on_samples)
        
        sample_layout.addWidget(detect_btn)
        sample_group.setLayout(sample_layout)
        
        # Results table
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        
        self.detection_table = QTableWidget()
        self.detection_info = QLabel("No detection results yet")
        
        results_layout.addWidget(self.detection_table)
        results_layout.addWidget(self.detection_info)
        results_group.setLayout(results_layout)
        
        layout.addWidget(sample_group)
        layout.addWidget(results_group)
    
    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Data", "", "CSV Files (.csv);;Excel Files (.xlsx *.xls)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    self.data = pd.read_excel(file_path)
                
                self.file_path_label.setText(os.path.basename(file_path))
                self.display_data()
                self.statusBar().showMessage(f"Data imported: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import data: {str(e)}")
    
    def load_sample_data(self):
        try:
            sample_path = 'sample_healthcare_claims.csv'
            if os.path.exists(sample_path):
                self.data = pd.read_csv(sample_path)
                self.file_path_label.setText(f"Sample data ({sample_path})")
                self.display_data()
                self.statusBar().showMessage("Sample data loaded")
            else:
                self.create_sample_data()
                self.data = pd.read_csv(sample_path)
                self.file_path_label.setText(f"Sample data ({sample_path})")
                self.display_data()
                self.statusBar().showMessage("Sample data created and loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load sample data: {str(e)}")
    
    def display_data(self):
        if self.data is None:
            return
        
        # Update info label
        self.data_info_label.setText(f"Rows: {self.data.shape[0]}, Columns: {self.data.shape[1]}")
        
        # Populate table
        self.data_table.clear()
        self.data_table.setRowCount(min(100, len(self.data)))  # Limit to 100 rows for display
        self.data_table.setColumnCount(len(self.data.columns))
        self.data_table.setHorizontalHeaderLabels(self.data.columns)
        
        # Fill table with data
        for i in range(min(100, len(self.data))):
            for j in range(len(self.data.columns)):
                value = str(self.data.iloc[i, j])
                item = QTableWidgetItem(value)
                
                # Highlight fraudulent claims in red
                if self.data.columns[j] == 'is_fraud' and value == '1':
                    item.setBackground(QColor(255, 200, 200))
                
                self.data_table.setItem(i, j, item)
        
        # Resize columns to content
        self.data_table.resizeColumnsToContents()
    
    def train_model(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please import data first")
            return
        
        if 'is_fraud' not in self.data.columns:
            QMessageBox.warning(self, "Warning", "Dataset must contain 'is_fraud' column")
            return
        
        try:
            # Prepare data
            # Drop non-numeric columns that won't work with the model
            X = self.data.drop('is_fraud', axis=1)
            
            # Handle non-numeric columns
            X = pd.get_dummies(X)
            
            y = self.data['is_fraud']
            
            # Store feature names
            self.features = X.columns
            
            # Split data
            test_size = self.test_size_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Start training thread
            n_estimators = self.n_estimators_spin.value()
            self.train_thread = ModelTrainingThread(
                self.X_train, self.X_test, self.y_train, self.y_test, n_estimators
            )
            
            # Connect signals
            self.train_thread.progress_update.connect(self.update_progress)
            self.train_thread.training_complete.connect(self.training_complete)
            
            # Start training
            self.train_thread.start()
            self.statusBar().showMessage("Training model...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train model: {str(e)}")
    
    def update_progress(self, value):
        self.train_progress.setValue(value)
    
    def training_complete(self, model, accuracy, report):
        # Store model
        self.model = model
        self.feature_importances = model.feature_importances_
        
        # Display results
        results_text = f"Model Training Results:\n\n"
        results_text += f"Accuracy: {accuracy:.4f}\n\n"
        results_text += f"Classification Report:\n{report}\n"
        
        self.results_text.setText(results_text)
        
        # Update feature importance table
        self.display_feature_importance()
        
        self.statusBar().showMessage("Model training completed")
    
    def display_feature_importance(self):
        if self.feature_importances is None:
            return
        
        # Sort features by importance
        indices = np.argsort(self.feature_importances)[::-1]
        
        # Display in table
        self.importance_table.setRowCount(len(indices))
        
        for i, idx in enumerate(indices):
            # Feature name
            feature_item = QTableWidgetItem(str(self.features[idx]))
            self.importance_table.setItem(i, 0, feature_item)
            
            # Importance value
            importance_item = QTableWidgetItem(f"{self.feature_importances[idx]:.4f}")
            self.importance_table.setItem(i, 1, importance_item)
            
            # Color high importance features
            if i < 5:  # Top 5 features
                feature_item.setBackground(QColor(200, 255, 200))
                importance_item.setBackground(QColor(200, 255, 200))
    
    def detect_fraud_on_samples(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please train a model first")
            return
        
        if self.X_test is None:
            QMessageBox.warning(self, "Warning", "No test data available")
            return
        
        try:
            # Make predictions on test data
            predictions = self.model.predict(self.X_test)
            probabilities = self.model.predict_proba(self.X_test)[:, 1]  # Probability of fraud
            
            # Get original indices from test set
            test_indices = self.y_test.index
            
            # Create results table
            self.detection_table.clear()
            self.detection_table.setRowCount(min(100, len(test_indices)))  # Limit to 100 rows
            self.detection_table.setColumnCount(6)  # Selected columns + prediction
            self.detection_table.setHorizontalHeaderLabels([
                "Claim ID", "Amount", "Provider", "Actual", "Predicted", "Probability"
            ])
            
            # Counter for correctly identified fraud
            correct_fraud = 0
            total_fraud = sum(self.y_test)
            displayed_rows = 0
            
            # Fill table with results
            for i, idx in enumerate(test_indices):
                if displayed_rows >= 100:
                    break
                    
                # Original row from dataset
                try:
                    claim_id = str(self.data.loc[idx, 'claim_id']) if 'claim_id' in self.data.columns else f"CLAIM-{idx}"
                except:
                    claim_id = f"CLAIM-{idx}"
                    
                try:
                    amount = f"${self.data.loc[idx, 'claim_amount']:.2f}" if 'claim_amount' in self.data.columns else "N/A"
                except:
                    amount = "N/A"
                    
                try:
                    provider = str(self.data.loc[idx, 'provider_id']) if 'provider_id' in self.data.columns else "N/A"
                except:
                    provider = "N/A"
                
                # Ground truth and prediction
                actual = self.y_test.iloc[i]
                predicted = predictions[i]
                probability = f"{probabilities[i]:.2%}"
                
                # Count correct fraud identifications
                if actual == 1 and predicted == 1:
                    correct_fraud += 1
                
                # Add to table
                self.detection_table.setItem(displayed_rows, 0, QTableWidgetItem(claim_id))
                self.detection_table.setItem(displayed_rows, 1, QTableWidgetItem(amount))
                self.detection_table.setItem(displayed_rows, 2, QTableWidgetItem(provider))
                self.detection_table.setItem(displayed_rows, 3, QTableWidgetItem(str(actual)))
                self.detection_table.setItem(displayed_rows, 4, QTableWidgetItem(str(predicted)))
                self.detection_table.setItem(displayed_rows, 5, QTableWidgetItem(probability))
                
                # Color code based on prediction correctness
                for col in range(6):
                    item = self.detection_table.item(displayed_rows, col)
                    if actual == 1 and predicted == 1:  # True positive (correctly identified fraud)
                        item.setBackground(QColor(255, 200, 200))
                    elif actual == 0 and predicted == 0:  # True negative
                        item.setBackground(QColor(200, 255, 200))
                    elif actual == 0 and predicted == 1:  # False positive
                        item.setBackground(QColor(255, 255, 200))
                    else:  # False negative (missed fraud)
                        item.setBackground(QColor(200, 200, 255))
                
                displayed_rows += 1
            
            # Resize columns to content
            self.detection_table.resizeColumnsToContents()
            
            # Update info label
            detection_rate = correct_fraud / total_fraud if total_fraud > 0 else 0
            self.detection_info.setText(
                f"Detected {correct_fraud} out of {total_fraud} fraudulent claims ({detection_rate:.2%})"
            )
            
            self.statusBar().showMessage("Fraud detection completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to detect fraud: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = SimpleFraudDetectionApp()
    window.show()
    sys.exit(app.exec_())

if _name_ == "_main_":
    main()
