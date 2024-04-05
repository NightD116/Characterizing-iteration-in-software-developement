import re
import string
import requests
import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import seaborn as sns
import joblib

# Function to fetch all pages of data
def fetch_all_pages(url, params={}, headers={}):
    all_data = []
    while url:
        response = requests.get(url,params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data)
            url = response.links.get('next', {}).get('url')  # Get URL for next page if available
        else:
            print(f"Error code {response.status_code}: Failed to fetch data from {url}")
            break
    return all_data

def get_data(owner, repo):
    per_page = 100
    # GitHub API endpoints
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    issues_url = f"{base_url}/issues?per_page={per_page}"
    events_url = f"{base_url}/issues?per_page={per_page}"
    releases_url = f"{base_url}/issues?per_page={per_page}"
    milestones_url = f"{base_url}/issues?per_page={per_page}"
    # Define the params and headers just care closed issues
    issues_milestone_params = {
        "state": "closed"
    }
    headers = {'User-Agent': 'request'}
    # fetch data and use dataframe to store relevant data
    issues_data = fetch_all_pages(url=issues_url, params=issues_milestone_params, headers=headers)
    issues_df = pd.DataFrame.from_records(issues_data, columns=["created_at","html_url", "name", "body", "published_at"])
    events_data = fetch_all_pages(url=events_url, headers=headers)
    events_df = pd.DataFrame.from_records(events_data, columns=["type", "payload"])
    releases_data = fetch_all_pages(url=releases_url, headers=headers)
    releases_df = pd.DataFrame.from_records(releases_data, columns=["created_at","html_url", "name", "body", "published_at"])
    milestones_data = fetch_all_pages(url=milestones_url, params=issues_milestone_params, headers=headers)
    milestones_df = pd.DataFrame.from_records(milestones_data, columns=["created_at","html_url", "title", "description", "open_issues", "closed_issues", "due_on", "closed_at"])
    return issues_df, events_df, releases_df, milestones_df

def detect_iterations(issues_df, events_df, releases_df, milestones_df):
    # Define the features of iteration
    iterations_df = pd.DataFrame(columns=["Type", "Related url","planned/unplanned", "Cause", "Body", "Start Time", "End Time", "Duration"])

    # Extract the release iteration
    for index,release in releases_df.iterrows():
        type = "Release"
        related_url = release.html_url
        body = "version: " + str(release.name) + ",\n Body:\n " +  str(release.body)
        isplanned = "planned"
        start_at = release.created_at
        end_at = release.published_at
        duration = pd.Timestamp(end_at) - pd.Timestamp(start_at)
        iterations_df.loc[len(iterations_df.index)] = [type, related_url, isplanned, "New Version publication", body, start_at, end_at, duration]
    # Extract the Milestone iteration
    for index,milestone in milestones_df.iterrows():
        type = "Milestone"
        milestone_url = milestone.html_url
        body = "Title: " + str(milestone.title) + ",\n Description:\n " +  str(milestone.description) + "\n open_issues: " + str(milestone.open_issues) + "\n closed_issues: " + str(milestone.closed_issues) + "\n due_on:" + str(milestone.due_on)
        isplanned = "planned"
        start_at = milestone.created_at
        end_at = milestone.closed_at
        duration = pd.Timestamp(end_at) - pd.Timestamp(start_at)
        iterations_df.loc[len(iterations_df.index)] = [type, related_url, isplanned, "New Version publication", body, start_at, end_at, duration]

    # Take list of reviewed pull request
    pull_request_review_events = events_df[events_df.type == "PullRequestReviewEvent"]
    pull_request_review_events_payload = pull_request_review_events["payload"]
    pr_review_payload_df = pd.DataFrame(list(pull_request_review_events_payload))
    pr_review_number_list = []
    for index, pr in pr_review_payload_df.iterrows():
        pr_review_number_list.append(pr["pull_request"]["number"])

    # Replace nan value in feature body by ""
    issues_df["body"].fillna("")
    # Detect Bug Fixing Process, Knowledge Integration, Direct Implementation
    for index, issue in issues_df.iterrows():
        related_url = issue.html_url
        body = str(issue.title)+ " " + str(issue.body)
        isplanned = "unplanned"
        start_at = issue.created_at  
        end_at = issue.closed_at
        duration = pd.Timestamp(end_at) - pd.Timestamp(start_at)
        if pd.isna(issue.pull_request):
            type = "Bug Fixing Process"
        else:
            if pd.notna(issue.milestone) or pd.notna(issue.assignee) or len(issue.assignees)!=0:
                isplanned = "planned"
            number = issue.number
            if number in pr_review_number_list:
                type = "Knowledge Integration"
            else:
                type = "Direct Implementation"
        iterations_df.loc[len(iterations_df.index)] = [type, related_url, isplanned, "Unknown", body, start_at, end_at, duration]
    return iterations_df

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

STOPWORDS = set(stopwords.words("english"))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r"", string)

def preprocess_data(text):
    text = clean_text(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = lemmatize_words(text)
    return text

def main():
    owner = "stanfordnlp"
    repo = "stanza"
    issues_df, events_df, releases_df, milestones_df = get_data(owner=owner, repo=repo)

    print("Get data Done!")
    
    iteration_df = detect_iterations(issues_df, events_df, releases_df, milestones_df)
    print("Detect Iteration Done!")
    
    models = joblib.load("trained_Random Forest_model.pkl")
    unknown_iterations = iteration_df[iteration_df["Cause"] == "Unknown"]
    
    X_pred = unknown_iterations["body"].astype(str).apply(preprocess_data)
    y_pred = models.predict(X_pred)
    
    iteration_df[iteration_df["Cause"] == "Unknown"]["Cause"] = y_pred
    iteration_df.to_csv(f"{owner}_{repo}_iterations_detected_data.csv")

if __name__ == "__main__":
    main()