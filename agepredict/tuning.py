import os
from functools import partial
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from agepredict.train import train_model

def tune_ray(cleaned_csv,img_dir, checkpoint_root="output/checkpoints", num_samples=10, max_epochs=20):

    # these are conservative tune choices since my system is not insanely powerful
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3, 1e-4),
        "batch_size": tune.choice([8, 16, 32, 64,128]),
        "early_stop_patience": tune.choice([2, 3, 4, 5]),
        "epochs": tune.choice([10, 15, 20, 25,30])
    }

    def train_ray(config, cleaned_csv=None, img_dir=None, checkpoint_root=None):

        trial_id = session.get_trial_id()
        check_path = os.path.join(checkpoint_root, f"ray_trial_{trial_id}.pth")

        final_rmse = train_model(
            csv_path=cleaned_csv,
            img_dir=img_dir,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            checkpoint_path=check_path,
            early_stop_patience=config["early_stop_patience"],
            step_size=5,   
            gamma=0.1
        )

        session.report({"rmse": final_rmse})

    trainable_with_args = partial(
        train_ray,
        cleaned_csv=str(cleaned_csv),
        img_dir=str(img_dir),
        checkpoint_root=str(checkpoint_root)
    )

    scheduler = ASHAScheduler(
        metric="rmse",
        mode="min",
        max_t=max_epochs,
        grace_period=3,
        reduction_factor=2
    )

    def short_dirname_creator(trial):
        return f"_{trial.trial_id}"

    run_path = os.path.abspath("age")

    analysis = tune.run(
        trainable_with_args,
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path=run_path,
        trial_dirname_creator=short_dirname_creator,
        name="age",
        resources_per_trial={"cpu": 4, "gpu": 1}
    )

    best_trial = analysis.get_best_trial(metric="rmse", mode="min")
    best_config = best_trial.config
    best_rmse = best_trial.last_result["rmse"]

    print("\nRay Tune Tuning Results")
    print(f"Best trial config: {best_config}")
    print(f"Best trial final RMSE: {best_rmse:.4f}")

    return best_config, best_rmse
