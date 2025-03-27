#!/usr/bin/env bash
# filepath: /Users/bolinw/Desktop/cmu/course_files/11695_mlip/milestones/ms1/group-project-s25-the-expendables/test_requests.sh

# Ensure the user_ids.txt file exists
if [ ! -f data/user_ids.txt ]; then
  echo "data/user_ids.txt not found! Please run get_user_ids.py first."
  exit 1
fi

# Read user_ids into an array using a while loop
user_ids=()
while IFS= read -r line; do
    # Skip empty lines
    if [ -n "$line" ]; then
        user_ids+=("$line")
    fi
done < data/user_ids.txt

# Check if the array is empty
if [ ${#user_ids[@]} -eq 0 ]; then
    echo "No user_ids found in data/user_ids.txt"
    exit 1
fi

end=$((SECONDS+1800))
while [ $SECONDS -lt $end ]; do
    random_index=$((RANDOM % ${#user_ids[@]}))
    user_id=${user_ids[$random_index]}
    # Send a GET request for the selected user_id
    curl "http://localhost:8082/recommend/$user_id"
    echo  # Newline after each response
    sleep 1
done