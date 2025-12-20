import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import cv2
from tqdm import tqdm
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set random seed for reproducibility
np.random.seed(42)

class PrescriptionDataAnalyzer:
    def __init__(self, data_dir, labels_path):
        """
        Initialize the data analyzer with data directory and labels path.
        
        Args:
            data_dir (str): Path to the directory containing training images
            labels_path (str): Path to the CSV file containing labels
        """
        self.data_dir = data_dir
        self.labels_path = labels_path
        self.df = None
        self.class_distribution = None
        
    def load_data(self):
        """Load and preprocess the dataset."""
        # Load labels
        self.df = pd.read_csv(self.labels_path)
        
        # Add full image paths
        self.df['image_path'] = self.df['IMAGE'].apply(
            lambda x: os.path.join(self.data_dir, 'training_words', x)
        )
        
        # Check which files actually exist
        self.df['exists'] = self.df['image_path'].apply(os.path.exists)
        
        # Filter out non-existent files
        self.df = self.df[self.df['exists']].drop('exists', axis=1)
        
        # Calculate class distribution
        self.class_distribution = self.df['GENERIC_NAME'].value_counts()
        
        return self.df
    
    def plot_class_distribution(self, top_n=20):
        """Plot the distribution of medicine classes."""
        if self.class_distribution is None:
            self.load_data()
            
        plt.figure(figsize=(12, 6))
        self.class_distribution.head(top_n).plot(kind='bar')
        plt.title(f'Top {top_n} Most Common Medicine Classes')
        plt.xlabel('Medicine Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def display_sample_images(self, n_samples=5):
        """Display sample images from each class."""
        if self.df is None:
            self.load_data()
            
        # Get one sample per class
        samples = self.df.groupby('GENERIC_NAME').apply(
            lambda x: x.sample(min(len(x), n_samples))
        ).reset_index(drop=True)
        
        # Display samples
        n_cols = min(5, n_samples)
        n_rows = int(np.ceil(len(samples) / n_cols))
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            try:
                img = Image.open(row['image_path'])
                plt.subplot(n_rows, n_cols, idx)
                plt.imshow(img, cmap='gray')
                plt.title(f"{row['GENERIC_NAME']}")
                plt.axis('off')
            except Exception as e:
                print(f"Error loading image {row['image_path']}: {e}")
        
        plt.tight_layout()
        plt.show()
    
    def get_image_stats(self, sample_size=100):
        """Get statistics about the images (dimensions, channels, etc.)."""
        if self.df is None:
            self.load_data()
            
        # Sample images for analysis
        sample = self.df.sample(min(sample_size, len(self.df)))
        
        stats = {
            'widths': [],
            'heights': [],
            'aspect_ratios': [],
            'channels': []
        }
        
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Analyzing images"):
            try:
                img = Image.open(row['image_path'])
                width, height = img.size
                stats['widths'].append(width)
                stats['heights'].append(height)
                stats['aspect_ratios'].append(width / height if height > 0 else 0)
                stats['channels'].append(len(img.getbands()))
            except Exception as e:
                print(f"Error processing {row['image_path']}: {e}")
        
        # Calculate statistics
        stat_summary = {
            'mean_width': np.mean(stats['widths']),
            'median_width': np.median(stats['widths']),
            'min_width': min(stats['widths']),
            'max_width': max(stats['widths']),
            'mean_height': np.mean(stats['heights']),
            'median_height': np.median(stats['heights']),
            'min_height': min(stats['heights']),
            'max_height': max(stats['heights']),
            'mean_aspect_ratio': np.mean(stats['aspect_ratios']),
            'channels': Counter(stats['channels'])
        }
        
        return stat_summary

class ImagePreprocessor:
    @staticmethod
    def resize_image(image, target_size=(224, 224)):
        """Resize image to target size while maintaining aspect ratio."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize with aspect ratio preserved
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with white background
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Paste the resized image onto the center of the white background
        new_image.paste(
            image,
            ((target_size[0] - image.size[0]) // 2,
             (target_size[1] - image.size[1]) // 2)
        )
        
        return new_image
    
    @staticmethod
    def preprocess_image(image_path, target_size=(224, 224)):
        """Preprocess a single image."""
        try:
            # Open image
            img = Image.open(image_path)
            
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)  # Increase contrast
            
            # Convert to numpy array for OpenCV operations
            img_array = np.array(img)
            
            # Apply adaptive thresholding
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Convert back to PIL Image
            img = Image.fromarray(img_array)
            
            # Resize
            img = ImagePreprocessor.resize_image(img, target_size)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None

class PrescriptionDataset(Dataset):
    def __init__(self, df, transform=None, preprocess=False):
        self.df = df
        self.transform = transform
        self.preprocess = preprocess
        self.label_to_idx = {label: idx for idx, label in enumerate(df['GENERIC_NAME'].unique())}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['GENERIC_NAME']
        
        # Load and preprocess image
        if self.preprocess:
            img = ImagePreprocessor.preprocess_image(img_path)
        else:
            img = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        # Convert label to tensor
        label_idx = self.label_to_idx[label]
        return img, label_idx

class PrescriptionClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PrescriptionClassifier, self).__init__()
        # Use ResNet18 as the base model
        self.model = models.resnet18(pretrained=pretrained)
        
        # Freeze the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=10):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    # Initialize paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'Training')
    labels_path = os.path.join(data_dir, 'training_labels.csv')
    
    # Initialize analyzer
    print("Initializing data analyzer...")
    analyzer = PrescriptionDataAnalyzer(data_dir, labels_path)
    
    # Load and explore data
    print("\nLoading and exploring data...")
    df = analyzer.load_data()
    print(f"Loaded {len(df)} images with {len(analyzer.class_distribution)} unique classes.")
    
    # Display class distribution
    print("\nClass distribution:")
    print(analyzer.class_distribution.head(10))
    print(f"\nTotal unique medicine names: {len(analyzer.class_distribution)}")
    
    # Plot class distribution
    print("\nPlotting class distribution...")
    analyzer.plot_class_distribution(top_n=15)
    
    # Display sample images
    print("\nDisplaying sample images...")
    analyzer.display_sample_images(n_samples=5)
    
    # Get image statistics
    print("\nAnalyzing image statistics...")
    stats = analyzer.get_image_stats()
    print("\nImage Statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    # Example of preprocessing a sample image
    print("\nExample of image preprocessing...")
    sample_image = df.iloc[0]['image_path']
    original = Image.open(sample_image)
    preprocessed = ImagePreprocessor.preprocess_image(sample_image)
    
    # Display original vs preprocessed
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Prepare data for training
    print("\nPreparing data for training...")
    
    # Filter classes with too few samples
    min_samples = 5
    value_counts = df['GENERIC_NAME'].value_counts()
    valid_classes = value_counts[value_counts >= min_samples].index
    df_filtered = df[df['GENERIC_NAME'].isin(valid_classes)]
    
    # Split data into train, validation, and test sets
    train_df, test_val_df = train_test_split(
        df_filtered, test_size=0.3, random_state=42, 
        stratify=df_filtered['GENERIC_NAME']
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.5, random_state=42,
        stratify=test_val_df['GENERIC_NAME']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    image_datasets = {
        'train': PrescriptionDataset(train_df, transform=data_transforms['train']),
        'val': PrescriptionDataset(val_df, transform=data_transforms['val']),
        'test': PrescriptionDataset(test_df, transform=data_transforms['test'])
    }
    
    # Create dataloaders
    batch_size = 32
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, 
                          shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size,
                        shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size,
                         shuffle=False, num_workers=4)
    }
    
    # Initialize the model
    print("\nInitializing the model...")
    num_classes = len(np.unique(df_filtered['GENERIC_NAME']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PrescriptionClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nTraining the model...")
    model = train_model(
        model, criterion, optimizer, dataloaders, 
        device, num_epochs=10
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    class_names = [image_datasets['train'].idx_to_label[i] for i in range(num_classes)]
    evaluate_model(model, dataloaders['test'], device, class_names)
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()
