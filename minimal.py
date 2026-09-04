from models import build_foris
from utils.visualization import visualize_prediction

ref_image_path, ref_mask_path = "assets/ref_cat_image.jpg", "assets/ref_cat_mask.png"
target_image_path = "assets/target_cat_image.jpg"
output_path = "target_cat_pred.png"

# Build model
model = build_foris()

# Set reference and target
model.set_reference(ref_image_path, ref_mask_path)
model.set_target(target_image_path)

# Predict
pred_mask = model.segment() 

# Save visualization
visualize_prediction(
  ref_image_path,
  ref_mask_path,
  target_image_path,
  pred_mask,
  output_path,
)