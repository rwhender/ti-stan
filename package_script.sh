rm dist/*
python3 setup.py sdist bdist_wheel
# python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/TIStan-0.1.2*
python3 -m twine upload dist/*
