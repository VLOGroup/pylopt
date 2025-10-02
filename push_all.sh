#!/bin/bash
set -e

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "You are on branch '$CURRENT_BRANCH', not main. Checkout main before running this script."
fi

echo "On branch main. Pushing to all remotes."
for remote in origin public; do
    git push "$remote" main
done

