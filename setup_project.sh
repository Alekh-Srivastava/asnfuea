#!/bin/bash

set -e

# Create directories
mkdir -p ml-project/{app/pages,data/raw,data/processed,notebooks,src/data,src/models,src/visualization,models,test}

# Create files
touch ml-project/app/{backend.py,Home.py}
touch ml-project/src/data/{__init__.py,processor.py,eda_util.py}
touch ml-project/src/models/{__init__.py,trainer.py,inference.py}
touch ml-project/src/visualization/__init__.py
touch ml-project/src/{__init__.py,main.py}
touch ml-project/create_environment.sh
touch ml-project/Dockerfile
touch ml-project/dev_requirements.txt
touch ml-project/setup.py
touch ml-project/README.md
touch ml-project/.gitignore

# Add __init__.py to all necessary folders
touch ml-project/src/data/__init__.py
touch ml-project/src/models/__init__.py
touch ml-project/src/visualization/__init__.py
touch ml-project/src/__init__.py

# Populate dev_requirements.txt
cat > ml-project/dev_requirements.txt <<EOL
black
jupyter-black
pylint
scikit-learn
pandas
numpy
matplotlib
seaborn
EOL

# Add pylint comment to Home.py
echo "# pylint: disable=invalid-name" > ml-project/app/Home.py

echo "Project template generated successfully."
