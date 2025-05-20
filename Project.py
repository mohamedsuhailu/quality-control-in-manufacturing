def _classify_defect(self, image, x, y, w, h, area):
        """Classify defect based on shape and intensity."""
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return "Unknown"

        aspect_ratio = w / h if h != 0 else 0
        mean_intensity = np.mean(roi)

        if aspect_ratio > 3:
            return "Scratch"
        elif area < 200:
            return "Spot"
        elif mean_intensity < 100:
            return "Dark Area"
        elif mean_intensity > 200:
            return "Light Area"
        else:
            return "Surface Defect"

# ================================
# 3. Save Annotated Results
# ================================
def save_results(vis_image, defects, original_image_name):
    """Save annotated image and defect report."""
    if not os.path.exists("qc_results"):
        os.makedirs("qc_results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(original_image_name)[0]
    image_output = f"qc_results/{base_name}_{timestamp}.jpg"
    text_output = f"qc_results/{base_name}_{timestamp}_defects.txt"

    # Save annotated image
    cv2.imwrite(image_output, vis_image)

    # Save defect report
    with open(text_output, 'w') as f:
        f.write(f"Defect Analysis for {original_image_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total defects: {len(defects)}\n\n")
        for defect in defects:
            f.write(f"Defect #{defect['id']}\n")
            f.write(f"  Type: {defect['type']}\n")
            f.write(f"  Area: {defect['area']:.2f} pixels\n")
            f.write(f"  Position: {defect['position']}\n\n")

    return image_output

# ================================
# 4. Run Complete QC Pipeline
# ================================
def run_qc():
    """Run the quality control inspection process."""
    print("Launching Quality Control System...\n" + "=" * 50)

    # Create or load images
    if not os.path.exists("qc_test_images") or not os.listdir("qc_test_images"):
        image_files = create_test_images()
    else:
        image_files = [f for f in os.listdir("qc_test_images") if f.lower().endswith(('.jpg', '.png'))]

    if not image_files:
        print("No test images found.")
        return

    reference_path = os.path.join("qc_test_images", image_files[0])
    qc_system = SimpleQCVision(reference_path)

    # Analyze each image
    for image_file in image_files[1:]:
        print(f"\nProcessing {image_file}...")
        image_path = os.path.join("qc_test_images", image_file)
        vis_image, defects = qc_system.detect_defects(image_path)

        if vis_image is not None:
            print(f"Detected {len(defects)} defect(s):")
            for defect in defects:
                print(f"  - ID #{defect['id']}: {defect['type']} | Area: {defect['area']:.2f}")

            output_path = save_results(vis_image, defects, image_file)
            print(f"Results saved: {output_path}")

            # Display results
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Defect Analysis - {image_file}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    print("\nQC inspection finished.")

# ================================
# 5. Main Entry Point
# ================================
if __name__ == "__main__":
    run_qc()
