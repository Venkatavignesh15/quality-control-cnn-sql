import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import sqlite3
import io
import datetime
import pandas as pd
from collections import defaultdict
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize database
def init_db():
    conn = sqlite3.connect("pcb_defects.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS defects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        image BLOB,
                        total_defects INTEGER,
                        avg_confidence REAL,
                        class_distribution TEXT
                      )''')
    conn.commit()
    conn.close()

# Convert image to BLOB
def image_to_blob(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# Save image to SQLite
def save_to_db(image, metrics):
    conn = sqlite3.connect("pcb_defects.db")
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    img_blob = image_to_blob(image)
    
    cursor.execute('''INSERT INTO defects 
                    (timestamp, image, total_defects, avg_confidence, class_distribution)
                    VALUES (?, ?, ?, ?, ?)''',
                 (timestamp, img_blob, 
                  metrics['total_defects'], 
                  metrics['avg_confidence'],
                  str(metrics['class_distribution'])))
    conn.commit()
    conn.close()

# Load stored images from the database
def load_images():
    conn = sqlite3.connect("pcb_defects.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, image, total_defects, avg_confidence, class_distribution FROM defects ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data

# Delete image from database
def delete_image(image_id):
    conn = sqlite3.connect("pcb_defects.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM defects WHERE id = ?", (image_id,))
    conn.commit()
    conn.close()

# Calculate metrics from YOLO results
def calculate_metrics(results):
    boxes = results[0].boxes
    return {
        'total_defects': len(boxes),
        'confidences': boxes.conf.tolist(),
        'class_distribution': {
            int(cls): count 
            for cls, count in zip(*np.unique(boxes.cls.cpu().numpy(), return_counts=True))
        },
        'avg_confidence': boxes.conf.mean().item() if len(boxes) > 0 else 0
    }

# Print terminal report
def print_terminal_report(metrics):
    print("\n" + "="*40)
    print("DEFECT DETECTION PERFORMANCE REPORT")
    print(f"Total Defects Found: {metrics['total_defects']}")
    print(f"Average Confidence: {metrics['avg_confidence']:.2%}")
    print("Class Distribution:")
    for cls_id, count in metrics['class_distribution'].items():
        print(f"- {model.names[cls_id]}: {count} defects")
    print("="*40 + "\n")

# Load YOLO model
model = YOLO("best.pt")

# Initialize database
init_db()

# --- Create Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detection", "📂 Stored Predictions", "📈 Performance Summary", "📊 Performance Evaluation"])

def evaluate_performance(test_images, test_labels):
    """Evaluate YOLO model performance on test data"""
    yolo_preds = []
    yolo_labels = []

    for img_path, label_path in zip(test_images, test_labels):
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_array = np.array(image)

        # Perform prediction
        results = model.predict(source=img_array, conf=0.1)
        
        # Convert YOLO results to binary (defect/no defect)
        has_defect = 1 if len(results[0].boxes) > 0 else 0
        yolo_preds.append(has_defect)
        
        # Load ground truth label
        yolo_labels.append(1 if os.path.exists(label_path) and os.path.getsize(label_path) > 0 else 0)

    # Calculate metrics
    accuracy = accuracy_score(yolo_labels, yolo_preds)
    precision = precision_score(yolo_labels, yolo_preds)
    recall = recall_score(yolo_labels, yolo_preds)
    f1 = f1_score(yolo_labels, yolo_preds)
    cm = confusion_matrix(yolo_labels, yolo_preds)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Defect', 'Defect'], 
                yticklabels=['No Defect', 'Defect'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    # Display metrics
    st.subheader("Performance Evaluation")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")


def load_test_data(test_dir):
    """Load test images and labels"""
    test_images = []
    test_labels = []

    # Get list of images and labels
    images_dir = os.path.join(test_dir, "images")
    labels_dir = os.path.join(test_dir, "labels")

    for img_file in os.listdir(images_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt"))
            test_images.append(img_path)
            test_labels.append(label_path)

    return test_images, test_labels

# --- Tab 1: Detection ---
with tab1:
    st.title("Product PCB Defect Detection")
    st.write("Upload an image to detect defects in your PCB.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        img_array = np.array(uploaded_image)

        # Perform prediction
        results = model.predict(source=img_array, conf=0.1)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Print to terminal
        print_terminal_report(metrics)
        
        # Generate annotated image
        result_img = results[0].plot()
        result_pil = Image.fromarray(result_img)

        # Save to database
        save_to_db(result_pil, metrics)

        # Display results
        st.subheader("Detection Results")
        
        # Metrics columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Defects", metrics['total_defects'])
        with col2:
            st.metric("Average Confidence", f"{metrics['avg_confidence']:.1%}")
        with col3:
            st.write("Class Distribution")
            for cls_id, count in metrics['class_distribution'].items():
                st.write(f"{model.names[cls_id]}: {count}")

        # Image comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Original Image")
        with col2:
            st.image(result_pil, caption="Annotated Prediction")
        
        st.success("Analysis complete! Results saved to database.")

# --- Tab 2: Stored Predictions ---
with tab2:
    st.title("📂 Stored Predictions")
    stored_data = load_images()

    if stored_data:
        cols = st.columns(3)
        for i, (img_id, timestamp, img_blob, defects, conf, cls_dist) in enumerate(stored_data):
            with cols[i % 3]:
                img = Image.open(io.BytesIO(img_blob))
                
                # Display metrics
                with st.expander(f"Prediction {img_id} - {timestamp}"):
                    st.image(img, use_column_width=True)
                    st.caption(f"Defects: {defects}")
                    st.caption(f"Avg Confidence: {conf:.1%}")
                    st.write("Class Distribution:")
                    cls_dist = eval(cls_dist)  # Convert string back to dict
                    for cls_id, count in cls_dist.items():
                        st.write(f"- {model.names[cls_id]}: {count}")
                    
                if st.button(f"Delete {img_id}", key=f"del_{img_id}"):
                    delete_image(img_id)
                    st.rerun()
    else:
        st.write("No stored predictions found.")

# --- Tab 3: Performance Summary ---
with tab3:
    st.title("📈 Performance Summary")
    
    # Load all records
    conn = sqlite3.connect("pcb_defects.db")
    df = pd.read_sql('''
        SELECT timestamp, total_defects, avg_confidence, class_distribution 
        FROM defects
        ORDER BY timestamp
    ''', conn)
    conn.close()
    
    if not df.empty:
        # Convert class distributions
        df['class_distribution'] = df['class_distribution'].apply(eval)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(df))
        col2.metric("Average Defects per Image", f"{df['total_defects'].mean():.1f}")
        col3.metric("Overall Confidence", f"{df['avg_confidence'].mean():.1%}")
        
        # Time series chart
        st.subheader("Defect Trend Over Time")
        df['date'] = pd.to_datetime(df['timestamp'])
        daily_stats = df.groupby(df['date'].dt.date).agg({
            'total_defects': 'sum',
            'avg_confidence': 'mean'
        }).reset_index()
        st.line_chart(daily_stats.set_index('date'))
        
        # Class distribution
        st.subheader("Overall Class Distribution")
        class_counts = defaultdict(int)
        for dist in df['class_distribution']:
            for cls_id, count in dist.items():
                class_counts[model.names[cls_id]] += count
        st.bar_chart(class_counts)
        
    else:
        st.write("No performance data available yet.")


# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the test folder
test_dir = os.path.join(current_dir, "test")

# --- Tab 4: Performance Evaluation ---
with tab4:
    st.title("📊 Model Performance Evaluation")
    
    # Load test data
    test_images, test_labels = load_test_data(test_dir)

    if st.button("Evaluate Model Performance"):
        evaluate_performance(test_images, test_labels)