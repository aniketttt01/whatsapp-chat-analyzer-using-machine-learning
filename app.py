import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Import the re module

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list = [user for user in user_list if user != "group_notification"]
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Top Statistics
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Messages", value=df.shape[0])
        with col2:
            st.metric(label="Total Words", value=df['message'].apply(lambda x: len(x.split())).sum())
        with col3:
            st.metric(label="Media Shared", value=df['message'].str.contains('<Media omitted>').sum())
        with col4:
            st.metric(label="Links Shared", value=df['message'].str.contains('http').sum())

        # Monthly Timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title('Activity Map')
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # Busiest Users
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Clustering for Common Words and Senders
        st.title("Common Words by Cluster and Sender")
        num_word_clusters = st.sidebar.slider("Number of Word Clusters", min_value=2, max_value=5, value=3,
                                              key="word_cluster_slider")
        word_cluster_counts, error_message_cluster = helper.cluster_messages_for_words(selected_user, df,
                                                                                       num_clusters=num_word_clusters)

        if error_message_cluster:
            st.warning(error_message_cluster)
        elif word_cluster_counts:
            for cluster_num, word_data in word_cluster_counts.items():
                st.subheader(f"Cluster {cluster_num + 1}")
                if word_data:
                    for word, senders in word_data.items():
                        sender_list = ", ".join([f"{sender} ({count})" for sender, count in senders])
                        st.markdown(f"- **{word}:** Sent by {sender_list}")
                else:
                    st.write("No common words found in this cluster.")

        # Spam Keyword Count Analysis
        st.title("Spam Keyword Usage")
        spam_keyword_counts, error_message_spam = helper.count_spam_keywords(selected_user, df)

        if error_message_spam:
            st.error(error_message_spam)
        elif spam_keyword_counts:
            for user, counts in spam_keyword_counts.items():
                st.subheader(f"User: {user}")
                for keyword, count in counts.items():
                    st.markdown(f"- Keyword '{keyword}': Used {count} times")
        else:
            st.info("No spam keywords found in the messages (based on spam.txt).")

        # Spam Detection (Basic) - Keep this if you still want the URL/user-defined keyword detection
        st.sidebar.header("Spam Detection (Basic)")
        url_threshold = st.sidebar.slider("Max URLs per message", min_value=1, max_value=10, value=3)
        spam_keywords_input = st.sidebar.text_input("Suspicious Keywords (comma-separated):")
        spam_keywords_list = [keyword.strip() for keyword in spam_keywords_input.split(',') if keyword.strip()]

        if st.sidebar.button("Detect Potential Spam"):
            st.title("Potential Spam Messages")
            spam_results_df = helper.detect_spam(selected_user, df.copy(), url_threshold, spam_keywords_list)  # Use a copy of df

            if not spam_results_df.empty:
                st.dataframe(spam_results_df)
                st.warning("These are potential spam messages based on basic rules. Accuracy may vary.")

                st.subheader("Keyword Usage in Potential Spam")
                keyword_usage = {}
                for keyword in spam_keywords_list:
                    keyword_usage[keyword] = {}
                    for index, row in spam_results_df.iterrows():
                        if keyword.lower() in row['message'].lower():
                            user = row['user']
                            keyword_usage[keyword][user] = keyword_usage[keyword].get(user, 0) + 1

                if keyword_usage:
                    for keyword, user_counts in keyword_usage.items():
                        st.markdown(f"**Keyword: '{keyword}'**")
                        if user_counts:
                            for user, count in user_counts.items():
                                st.markdown(f"- {user}: {count} times")
                        else:
                            st.markdown("- Not used in flagged messages.")
                else:
                    st.info("No suspicious keywords found in the flagged messages.")

            else:
                st.success("No potential spam messages found based on the current rules.")