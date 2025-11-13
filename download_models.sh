#!/bin/bash

# -----------------------------------------------------------------
# >> CONFIGURATION (ADD MODEL URLS HERE) <<
# -----------------------------------------------------------------
#
# Add the direct download links for any GGUF models you want.
# To find a link:
# 1. Go to the model's page on Hugging Face (e.g., "Phi 3.5 Mini GGUF").
# 2. Click on the "Files and versions" tab.
# 3. Find the GGUF file you want (e.g., "Q4_K_M.gguf").
# 4. Click the small "download" icon (arrow pointing down) to the right.
# 5. Right-click the "download" button on the *next* page and "Copy Link Address".
#
declare -a MODEL_URLS=(
  "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true"
  "https://huggingface.co/google/gemma-2-9b-it-gguf/resolve/main/gemma-2-9b-it-q4_k_m.gguf?download=true"
)

# -----------------------------------------------------------------
# >> DOWNLOAD SCRIPT <<
# (You shouldn't need to edit below this line)
# -----------------------------------------------------------------

# The folder to download models into (matches run_test.sh)
TARGET_DIR="./llm-models"

# Create the directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "--- Starting Model Download ---"
echo "Target directory: $TARGET_DIR"
echo ""

for url in "${MODEL_URLS[@]}"; do
  # Extract the filename from the URL (strips the ?... part)
  filename=$(basename "${url%?*}")
  filepath="$TARGET_DIR/$filename"

  # Check if the file already exists
  if [ -f "$filepath" ]; then
    echo "-> [SKIPPED] File already exists: $filename"
  else
    echo "-> [DOWNLOADING] $filename"
    # Use curl: 
    # -L (follow redirects, which Hugging Face uses)
    # -o (specify output file)
    # -# (show a simple progress bar)
    curl -L -# -o "$filepath" "$url"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
      echo "-> [SUCCESS] Downloaded $filename"
    else
      echo "-> [FAILED] Download failed for $filename"
      # Optional: remove the partial file on failure
      # rm -f "$filepath"
    fi
  fi
  echo "" # Add a blank line for readability
done

echo "--- Model Download Complete ---"