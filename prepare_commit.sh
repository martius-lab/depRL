poetry run pre-commit run --all-files

cd docs
poetry run make html
cd ..

poetry run pytest tests/
