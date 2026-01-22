import os
import cv2
import numpy as np

def create_dummy_data(root='data'):
    for phase in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(root, phase, label)
            os.makedirs(path, exist_ok=True)
            
            # Create 5 dummy images per class per phase
            for i in range(5):
                # Random noise image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(path, f'dummy_{i}.jpg'), img)
    print("Dummy data created.")

if __name__ == '__main__':
    create_dummy_data()
