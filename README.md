# Forecasting-Tornadoes

This repository [^quick_note] contains statistical models for forecasting tornadoes.

[^quick_note]: Lightweight repositories for forecasting and inference, such as this one (usually created by @AFg6K7h4fhy2), are typically made as experimental test units for tackling Metaculus questions with statistical modeling or for the sake of the collaborators' understanding of statistical methods.

If you want to collaborate and create a model, please clone this repository via:

`git clone git@github.com:AFg6K7h4fhy2/Forecasting-Tornadoes.git`

and then create a new branch with your model `git checkout -b <your_branch>`. If you have feedback for the authors of this repository, please [make an issue](https://github.com/AFg6K7h4fhy2/Forecasting-Tornadoes/issues).

This repository uses `poetry` for dependency management. To install poetry, you can run `pipx install poetry` [^poetry]. And to activate the environment for this repository, run `poetry install`. If you would like to learn more about where `poetry` is installing the dependencies associated with this project, please run `poetry env info --path`.

[^poetry]: More about `poetry` can be learned on their [website](https://python-poetry.org/).

The canonical structure for building and documenting a model in this repository is as follows:

* Create a branch with your model name: `git checkout -b model_example_01`
* Create a new folder in the `models` directory: `cd model`, `mkdir model_example_01`, `cd model_example_01`
* Within your model directory, create `src`, `test`, and `out` folder, and a `params.toml` file.
  * Within `src`, create a `run.py` and `model.py` file, where `run.py` performs [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load) tasks, model execution, and post-processing tasks and `model.py` contains the model, which will typically be written in NumPyro. Output should be written to `out`. You can write an additional `utils`.
  * The optional `test` folder should contain `tests` written via `pytest`.
  * The `params.toml` file should contain parameters and values used in the modeling process (this is a config file).
* Write a model description in `./website/posts`. Follow existing examples for the `yaml` header of the `.qmd` file.
* Update your edits and modifications (hopefully not in a single commit!): `git add -A`; `git commit -m <message>`; `git push -u origin <model_example_01>`. To have your model on `main`, please make a pull request.

<!-- ## Repository Structure

`tree | grep -Ev "\.png|\.pyc|\.txt|\.csv"` -->
