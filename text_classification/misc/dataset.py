from __future__ import print_function
import json
import time
import csv
import praw

# Replace below with information provided to you by Reddit when registering your script
reddit = praw.Reddit(client_id="ekaQEDeIAyIkYeQjMBIKpw",
                     client_secret="5g8xpbJypoTkf2ywanevOCGN7xkJWQ",
                     user_agent="qa/1.0 (by /u/gateluv)")

# Initialize list to hold extracted data
extracted_data = []

# Counter for the number of posts processed
post_count = 0

# Define the start and end of the desired range
start_range = 95001
end_range =   100000

with open('coarse_discourse_dataset.json') as jsonfile:
    for line in jsonfile:
        # Check if we've reached the end of the desired range
        if post_count >= end_range:
            break

        try:
            reader = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

        # Check if the URL contains 'comments'
        if 'comments' not in reader['url']:
            continue

        submission = reddit.submission(url=reader['url'])

        # Annotators only annotated the 40 "best" comments determined by Reddit
        submission.comment_sort = 'best'
        submission.comment_limit = 40

        post_id_dict = {}

        for post in reader['posts']:
            post_id_dict[post['id']] = post

        try:
            full_submission_id = 't3_' + submission.id
            if full_submission_id in post_id_dict:
                post_count += 1  # Increment post_count for the submission itself
                if start_range <= post_count <= end_range:
                    post = post_id_dict[full_submission_id]
                    data_row = {
                        'url': submission.url,
                        'author': submission.author.name if submission.author else None,
                        'content': submission.selftext,
                        'main_type': post['majority_type'] if 'majority_type' in post else None
                    }
                    extracted_data.append(data_row)

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                full_comment_id = 't1_' + comment.id
                if full_comment_id in post_id_dict:
                    post_count += 1  # Increment post_count for every comment processed
                    if start_range <= post_count <= end_range:
                        post = post_id_dict[full_comment_id]
                        data_row = {
                            'url': submission.url,
                            'author': comment.author.name if comment.author else None,
                            'content': comment.body,
                            'main_type': post['majority_type'] if 'majority_type' in post else None
                        }
                        extracted_data.append(data_row)

        except Exception as e:
            print('Error %s' % (e))

        # To keep within Reddit API limits
        time.sleep(2)

# Append the extracted data to a CSV file
with open('extracted_data.csv', 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['url', 'author', 'content', 'main_type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Check if the file is empty to decide on writing headers
    csvfile.seek(0, 2)  # Move to the end of the file
    if csvfile.tell() == 0:  # If file is empty, write headers
        writer.writeheader()

    for row in extracted_data:
        writer.writerow(row)
