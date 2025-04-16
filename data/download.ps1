# Enable error handling and verbose output
$ErrorActionPreference = "Stop"

# Create and enter the "test" folder
# New-Item -ItemType Directory -Path "test" -Force
cd "test"
Start-BitsTransfer -Source "https://storage.googleapis.com/searchless_chess/data/test/action_value_data.bag" -Destination "action_value_data.bag"
Start-BitsTransfer -Source "https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag" -Destination "behavioral_cloning_data.bag"
Start-BitsTransfer -Source "https://storage.googleapis.com/searchless_chess/data/test/state_value_data.bag" -Destination "state_value_data.bag"
cd ..

# Create and enter the "train" folder
New-Item -ItemType Directory -Path "train" -Force
cd "train"

# Loop to download numbered train files
for ($idx = 0; $idx -lt 3; $idx++) {
    $formattedIdx = "{0:D5}" -f $idx  # Format as 00000, 00001, etc.
    $url = "https://storage.googleapis.com/searchless_chess/data/train/action_value-$formattedIdx-of-02148_data.bag"
    Start-BitsTransfer -Source $url -Destination "action_value-$formattedIdx-of-02148_data.bag"
}

# Download remaining train files
Start-BitsTransfer -Source "https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag" -Destination "behavioral_cloning_data.bag"
Start-BitsTransfer -Source "https://storage.googleapis.com/searchless_chess/data/train/state_value_data.bag" -Destination "state_value_data.bag"
cd ..
