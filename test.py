import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

directory = 'Data/'

# Get ALL class names
class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

print(f"Found {len(class_names)} classes: {class_names}")

# Calculate grid size for all classes
num_classes = len(class_names)
cols = 6  # Number of columns in the grid
rows = math.ceil(num_classes / cols)  # Calculate needed rows

# Create figure with dynamic size based on number of classes
fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
axes = axes.ravel() if rows > 1 else [axes]  # Handle single row case

# Process each class
for idx, class_name in enumerate(class_names):
    class_path = os.path.join(directory, class_name)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found for class {class_name}")
        axes[idx].axis('off')
        continue
    
    # Take first image
    sample_image_path = os.path.join(class_path, image_files[0])
    img = cv2.imread(sample_image_path)
    
    if img is None:
        print(f"Could not read image: {sample_image_path}")
        axes[idx].axis('off')
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_landmarks = img_rgb.copy()  # Create a copy to draw on
    
    # Process and draw landmarks
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_with_landmarks,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    axes[idx].imshow(img_with_landmarks)
    axes[idx].set_title(f'{class_name}\n({len(image_files)} images)')
    axes[idx].axis('off')

# Hide any empty subplots
for idx in range(len(class_names), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.suptitle(f'ASL Classes Visualization - All {len(class_names)} Classes', fontsize=16, y=1.02)
plt.show()

# Print summary
print(f"\nVisualized {len(class_names)} classes:")
for class_name in class_names:
    class_path = os.path.join(directory, class_name)
    image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  {class_name}: {image_count} images")