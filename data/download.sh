
set -ex

mkdir test
cd test
wget https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag
cd ..

mkdir train
cd train
for idx in $(seq -f "%05g" 0 100)
do
  wget https://storage.googleapis.com/searchless_chess/data/train/action_value-$idx-of-02148_data.bag
done
wget https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag
wget https://storage.googleapis.com/searchless_chess/data/train/state_value_data.bag
cd ..
