FILE_ORG="./dataset/cifar-100-python.tar.gz"
FILE_COR="./dataset/CIFAR-100-C.tar"

# download cifar-100
if [[ -f "$FILE_ORG" ]]; then
  echo "$FILE_ORG exists."
else
  echo "$FILE_ORG does not exist. Start downloading..."
  if [[ ! -d "./dataset" ]]; then
    mkdir dataset
  fi
  cd dataset
  wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
fi

# download cifar-100-c
if [[ -f "$FILE_COR" ]]; then
  echo "$FILE_COR exists. Start processing..."
else
  echo "$FILE_COR does not exist. Start downloading..."
  cd dataset
  wget https://zenodo.org/api/files/8fafaa0e-d7e5-448b-a5af-e8b5e1bd24ce/CIFAR-100-C.tar
  echo "Download succeeded. Start processing..."
fi

cd dataset
# unzip downloaded files
tar -zxvf cifar-100-python.tar.gz
tar -xvf CIFAR-100-C.tar

# process data
cd ..
python process_cifar.py cifar-100c

# for CIFAR-100-C, move original data to "severity-all"
cd dataset/CIFAR-100-C
if [[ ! -d "./corrupted/severity-all" ]]; then
  mkdir ./corrupted/severity-all
fi
mv *.npy ./corrupted/severity-all
