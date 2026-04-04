#!/bin/bash

# Ensure the pretrained directory is completely ignored by Git 
# so heavy model weights are never accidentally pushed
if ! grep -q "^pretrained/" .gitignore 2>/dev/null; then
    echo "pretrained/" >> .gitignore
    echo "Added pretrained/ to .gitignore"
fi

# Add all files (the .gitignore will automatically prevent the pretrained folder from being added)
git add .

# Commit your changes
git commit -m "feat: integrate Tiny ImageNet diffusion pipeline with UI and loss tracking"

# Push to the remote repository
git push origin HEAD

echo "Push complete!"