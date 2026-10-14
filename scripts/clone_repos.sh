# Partial (blob-less) bare clones of the 18 SWE-bench repositories into data/repos/.
# Optional: the evaluation harness clones any missing repository on first use.
set -e
REPOS_DIR="$(dirname "$0")/../data/repos"
mkdir -p "$REPOS_DIR"
cd "$REPOS_DIR"
for r in marshmallow-code/marshmallow pvlib/pvlib-python pydicom/pydicom pylint-dev/astroid pyvista/pyvista sqlfluff/sqlfluff psf/requests pallets/flask mwaskom/seaborn pydata/xarray pylint-dev/pylint pytest-dev/pytest astropy/astropy sphinx-doc/sphinx scikit-learn/scikit-learn matplotlib/matplotlib sympy/sympy django/django; do
  d="${r//\//__}.git"
  [ -d "$d" ] || git clone --bare --filter=blob:none "https://github.com/$r.git" "$d"
  echo "done $r"
done
