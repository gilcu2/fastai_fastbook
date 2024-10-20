from fastai.vision.all import *
from fastai.learner import load_learner
import typer
import pathlib

app = typer.Typer()

models_path = "data/models"
cat_dog_name = "cat_dog_classifier"
home_dir = str(Path.home())
project_dir = pathlib.Path(__file__).parent.parent.resolve()
models_dir = project_dir / models_path


def is_cat(x): return x[0].isupper()


@app.command()
def tune4_cat_dog(model_name: str = cat_dog_name):
    path = untar_data(URLs.PETS) / 'images'
    data_loaders = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224))
    classifier = vision_learner(data_loaders, resnet34, metrics=error_rate)
    classifier.fine_tune(1)

    classifier.path = Path(models_dir)
    classifier.export(f"{model_name}.pkl")
    export_path=f"{models_dir}/{model_name}.pkl"
    return classifier, export_path


@app.command()
def classifiy_cat_dog(image_path: str = "data/images/cat.jpg", model_name: str = cat_dog_name):
    img = PILImage.create(image_path)
    model_path = Path(f"{models_path}/{model_name}.pkl")
    classifier = load_learner(model_path, cpu=False)
    was_cat, _, probs = classifier.predict(img)
    print(f"Is this a cat?: {was_cat}.")
    print(f"Probability it's a cat: {probs[1].item():.6f}")
    return was_cat, probs[1].item()


if __name__ == "__main__":
    app()
