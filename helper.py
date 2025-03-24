from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re
from collections import Counter

def count_spam_keywords(selected_user, df, spam_file="spam.txt"):
    """Counts the occurrences of keywords from a spam file per user."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    try:
        with open(spam_file, 'r') as f:
            spam_keywords = [line.strip().lower() for line in f]
    except FileNotFoundError:
        return None, f"Error: Spam keyword file '{spam_file}' not found."

    keyword_counts_per_user = {}
    for index, row in df.iterrows():
        message = row['message'].lower()
        user = row['user']
        for keyword in spam_keywords:
            if keyword in message:
                if user not in keyword_counts_per_user:
                    keyword_counts_per_user[user] = Counter()
                keyword_counts_per_user[user][keyword] += 1

    return keyword_counts_per_user, None
plt.rcParams['font.family'] = 'Segoe UI Emoji'

extract = URLExtract()

def fetch_stats(selected_user,df):
    """Fetches basic statistics of the chat."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    """Identifies the most active users in the chat."""
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df_percent

def create_wordcloud(selected_user,df):
    """Generates a word cloud of the most frequent words."""
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    """Finds the most common words in the chat."""
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    """Analyzes the usage of emojis in the chat."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):
    """Generates the monthly timeline of messages."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):
    """Generates the daily timeline of messages."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):
    """Shows the count of messages for each day of the week."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    """Shows the count of messages for each month."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    """Generates a heatmap of activity based on day of the week and period of the day."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def cluster_messages_for_words(selected_user, df, num_clusters=3): # Reduced default clusters
    """Clusters messages and returns the count of most common words per cluster with senders."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    text_messages_df = df[~df['message'].isin(['<Media omitted>', ''])]
    text_messages = text_messages_df['message']

    if text_messages.empty:
        return None, "No text messages available for clustering."

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_messages)

    if tfidf_matrix.shape[0] < num_clusters:
        return None, f"Number of text messages ({tfidf_matrix.shape[0]}) is less than the requested number of clusters ({num_clusters})."

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)

    df_clustered = text_messages_df.copy()
    df_clustered['cluster'] = clusters

    cluster_word_counts = {}
    for cluster_num in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_num]
        all_words = []
        for message in cluster_data['message']:
            all_words.extend(message.lower().split())
        stop_words_file = open('stop_hinglish.txt', 'r')
        stop_words = stop_words_file.read().splitlines()
        filtered_words = [word for word in all_words if word not in stop_words and word.isalnum()]
        most_common = Counter(filtered_words).most_common(10) # Get top 10 common words

        word_sender_counts = {}
        for word, count in most_common:
            sender_counts = Counter(cluster_data[cluster_data['message'].str.contains(r'\b' + re.escape(word) + r'\b', case=False)]['user'])
            word_sender_counts[word] = sender_counts.items()

        cluster_word_counts[cluster_num] = word_sender_counts

    return cluster_word_counts, None