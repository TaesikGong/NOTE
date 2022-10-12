FILE_ORG="./dataset/cifar-10-python.tar.gz"
FILE_COR="./dataset/CIFAR-10-C.tar"

# download cifar-10
if [[ -f "$FILE_ORG" ]]; then
  echo "$FILE_ORG exists."
else
  echo "$FILE_ORG does not exist. Start downloading..."
  if [[ ! -d "./dataset" ]]; then
    mkdir dataset
  fi
  cd dataset
  wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
fi

# download cifar-10-c
if [[ -f "$FILE_COR" ]]; then
  echo "$FILE_COR exists. Start processing..."
else
  echo "$FILE_COR does not exist. Start downloading..."
  cd dataset
  wget https://zenodo.org/api/files/a35f793a-6997-4603-a92f-926a6bc0fa60/CIFAR-10-C.tar
  echo "Download succeeded. Start processing..."
fi

cd dataset
# unzip downloaded files
tar -zxvf cifar-10-python.tar.gz
tar -xvf CIFAR-10-C.tar

# process data
cd ..
python process_cifar.py cifar-10c

# for CIFAR-10-C, move original data to "severity-all"
cd dataset/CIFAR-10-C
if [[ ! -d "./corrupted/severity-all" ]]; then
  mkdir ./corrupted/severity-all
fi
mv *.npy ./corrupted/severity-all