# Explainable Pneumonia Detection — Legacy Project Wrapper

This repository is a legacy wrapper that links to the pneumonia-detection project as a Git submodule.

## Current Structure

```text
.
|-- .gitmodules
|-- Explainable-AI-Pneumonia-Detection-project/
`-- README.md
```

The submodule URL redirects to:

[guru8880/pneumonia-detection-using-ML](https://github.com/guru8880/pneumonia-detection-using-ML)

That repository contains the maintained CNN, SHAP, and LIME implementation and should be used for project details, setup, and future development.

## Clone with the Submodule

```bash
git clone --recurse-submodules https://github.com/guru8880/project-phase.git
cd project-phase
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Portfolio Note

Keeping two public repositories for the same project can confuse visitors and split project history. Consider archiving this wrapper and directing users to the maintained pneumonia-detection repository.
