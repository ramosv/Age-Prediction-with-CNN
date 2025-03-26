from pathlib import Path
from agepredict  import tune_ray, generate_submission, train_model, cross_val_training
import pandas as pd

def main():
    root = Path("C:/Users/ramos/Desktop/GitHub/Kaggle-Competition-Age-Prediction/age-prediction-spring-25-at-cu-denver")
    cleaned_csv = root / "wiki_labels_clean.csv"
    img_dir = root / "wiki_labeled/wiki_labeled"
    checkpoint_root = root / "output/checkpoints"
    checkpoint = checkpoint_root / "best_ray_model_val.pth"

    #hyperparameter tunnning with ray  
    best_config, best_rmse = tune_ray(
        cleaned_csv=str(cleaned_csv),
        img_dir=str(img_dir),
        checkpoint_root=str(checkpoint_root),
        num_samples=10,   
        max_epochs=50
    )
    print(f"Best config from Ray Tune: {best_config}")
    print(f"Best RMSE from Ray Tune: {best_rmse}")

    #cross validation tranining. takes a while to run maybe reduce n splits
    avg_rmse = cross_val_training(
        csv_path=str(cleaned_csv),
        img_dir=str(img_dir),
        n_splits=5,
        epochs=best_config["epochs"],
        batch_size=best_config["batch_size"],
        lr=best_config["lr"],
    )
    print(f"Cross-validation done. Average RMSE: {avg_rmse}")


    #Terminal outputs from ray tune. the raw terminal outputs has this info as well just copied it here to make it easy to test
    
    #Best config from Ray Tune: {'lr': 0.00022165285731130848, 'batch_size': 16, 'early_stop_patience': 3, 'epochs': 20}
    #Best trial config: {'lr': 0.00013065143261376387, 'batch_size': 64, 'early_stop_patience': 4, 'epochs': 30}
    #train_fn_ray_b8a1d_00009 TERMINATED 0.0002169 128 4 15 1 687.336 6.9934 
    
    best_config = {
        "lr": 0.000186,
        "batch_size": 128,
        "early_stop_patience": 6,
        "epochs": 20
    }

    final = train_model(
        csv_path=str(cleaned_csv),
        img_dir=str(img_dir),
        epochs=best_config["epochs"],
        batch_size=best_config["batch_size"],
        lr=best_config["lr"],
        checkpoint_path=str(checkpoint)
    )
    print(f"Final model RMSE: {final}")
 
    judge_csv = root / "wiki_judge.csv"
    judge_img_dir = root / "wiki_judge_images/wiki_judge_images"
    submission_path = root / "output/submission.csv"

    generate_submission(
        judge_csv=judge_csv,
        judge_img_dir=judge_img_dir,
        checkpoint_path=str(checkpoint),
        submission_path=str(submission_path)
    )
    print(f"Saved submission at: {submission_path}")

if __name__ == "__main__":
    root = Path("C:/Users/ramos/Desktop/GitHub/Kaggle-Competition-Age-Prediction/age-prediction-spring-25-at-cu-denver/output/submission.csv")
    roundtoone = pd.read_csv(root)
    #round age column to one decimal place
    roundtoone['age'] = roundtoone['age'].round(1)
    print(roundtoone)
    
    #save to csv
    roundtoone.to_csv("last_submission_rounded.csv", index=False)
    print(f"Submission file saved at: {root}")

    main()
