import os
import subprocess

def run_step(description, command):
    print(f"\n{'='*70}\n➡️ {description}\n{'='*70}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error during: {description}")
        exit(1)
    else:
        print(f"✅ Completed: {description}")

def main():
    # Step 1: Download dataset (if needed)
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        run_step("Downloading dataset from Kaggle", "python src/download_kaggle.py --ref paultimothymooney/chest-xray-pneumonia --out data")
    else:
        print("📦 Data folder already exists. Skipping download.")

    # Step 2: Preprocess dataset
    run_step("Preprocessing dataset", "python src/preprocessing.py")

    # Step 3: Train the model
    run_step("Training model", "python src/train.py")

    # Step 4: Generate explanations
    run_step("Generating explanations (SHAP & LIME)", "python src/explain.py")

    # Step 5: Evaluate model
    run_step("Evaluating model performance", "python src/evaluate.py")

    print("\n🎯 Project pipeline completed successfully!")

if __name__ == "__main__":
    main()
