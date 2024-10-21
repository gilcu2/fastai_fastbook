from pathlib import Path
from fastai.vision.all import Learner

def learner_export(learner: Learner ,models_dir:str,model_name:str):
    learner.path = Path(models_dir)
    learner.export(f"{model_name}.pkl")
    export_path = f"{models_dir}/{model_name}.pkl"