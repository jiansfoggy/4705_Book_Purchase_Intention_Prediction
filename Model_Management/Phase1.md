# Book Purchase Intention Prediction

## Phase 1: Experimentation and Model Management

1. run `train_model.py` to start training. The model we use here is FunkSVD.

```bash
python3 -m pip install wandb
wandb login # copy and paste API Key while asking
export WANDB_API_KEY=<WANDB_API_KEY>
python3 train_model.py
```
2. The model weight is saved in the current directory.
3. **Weights & Biases** (WandB) helps log all essential information, like Git Commit, hyperparameters, performance metrics, and data version.

   Please click [here](https://wandb.ai/jsfoggy/Book_Purchase_Intention_Prediction/table?nw=nwuserjsfoggy) to check the previous records.

4. WandB also helps save 3 artifacts such as **dataset** artifact, **model** one, and **code** one via the following functions.

    ```python
    # 1.Create code artifact
    if save_code:
        try:
            # log_code will capture python files in repo as an artifact
            run.log_code(".")
        except Exception as e:
            # best-effort: continue even if code snapshot fails
            run.log({"_code_logging_error": str(e)})

    # 2.Create data artifact
    data_art = wandb.Artifact(name=f"{dataset_name}-artifact", type="dataset", metadata={})
    for p in data_paths:
        if os.path.exists(p):
            data_art.add_file(p)
    run.log_artifact(data_art)
    data_art.wait()
    data_art.aliases.append("staging")
    
    # 3.Create model artifact
    model_art = wandb.Artifact(name=f"{model_name}-artifact", type="model", metadata={})
    model_art.add_file(model_path)
    run.log_artifact(model_art)
    ```
5. Then, the following code finishes **Model Registry** and promote best-performing model to a "production".

    ```python
    # Link to Model Registry
    alias = ["production"]
    run.link_model(path=model_path, 
                   registered_model_name=f"{model_name}-artifact", 
                   aliases=[alias])
    ```
    