# Get AIMaster version from setup/version.py
set -o errexit

current_path=$(realpath .)
version_file=${current_path}/deeprec_master/__init__.py
version=$(cat $version_file | grep "__version__" | awk '{print $3}' | head -n 1 | sed 's/\"//g')
if [[ -z ${version} ]]; then
  echo "ERROR: not found aimaster version from setup/version.py"
  exit -1
fi

echo $version
