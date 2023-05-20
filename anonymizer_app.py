import streamlit as st
import pandas as pd
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime
import wordmap
import replace_with_synonym

st.title("Data Anonymizer")

# Load the Twitter data CSV file
uploaded_file = st.file_uploader("Upload Tweets dataset", type="csv")
if uploaded_file is not None:
    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(uploaded_file, nrows=20)
    # Show the original data table
    st.write("Original data table")
    st.write(df)

    # Define the regex pattern in a Presidio `Pattern` object:
    pan_number_pattern = Pattern(name="pan_number_pattern", regex="[A-Z]{5}[0-9]{4}[A-Z]{1}", score=0.5)

    # Define the recognizer with one or more patterns
    pan_number_recognizer = PatternRecognizer(supported_entity="PAN_NUMBER", patterns=[pan_number_pattern])

        # Define the regex pattern in a Presidio `Pattern` object:
    twitter_mention_pattern = Pattern(name="twitter_mention_pattern", regex="@(\\w+)", score=0.5)

    # Define the recognizer with one or more patterns
    twitter_mention_recognizer = PatternRecognizer(supported_entity="TWITTER_MENTION", patterns=[twitter_mention_pattern])

    # # Analyzer output
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(pan_number_recognizer)
    analyzer.registry.add_recognizer(twitter_mention_recognizer)


    # Anonymizer output
    # anonymizer = AnonymizerEngine()

    # Define anonymization operators
    # operators = {
    #     "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
    #     "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
    #     "PHONE_NUMBER": OperatorConfig(
    #         "mask",
    #         {
    #             "type": "mask",
    #             "masking_char": "*",
    #             "chars_to_mask": 7,
    #             "from_end": True,
    #         },
    #     ),
    #     "EMAIL_ADDRESS": OperatorConfig(
    #         "mask",
    #         {
    #             "chars_to_mask": 7,
    #             "type": "mask",
    #             "from_end": False,
    #             "masking_char": "*"
    #         }
    #     ),
    #     "TITLE": OperatorConfig("redact", {}),
    # }

    # Create a new column "Anonymized Tweet" with the anonymized tweet
    anonymized_tweets = []
    for tweet in df["Tweet Content"]:
        print("original: ", tweet)
        analyzer_results = analyzer.analyze(text=tweet, language="en")
        for result in analyzer_results:
            # print(tweet[result.start : result.end])
            tweet = tweet.replace(tweet[result.start : result.end], wordmap.returnString(tweet[result.start : result.end]))
            tweet = replace_with_synonym.replace_adjectives_with_synonyms(tweet)

        print("wordmapped: ", tweet)
        # anonymized_results = anonymizer.anonymize(
        #     text=tweet, analyzer_results=analyzer_results, operators=operators
        # )
        # anonymized_tweet = anonymized_results.text
        # anonymized_tweets.append(anonymized_tweet)
        anonymized_tweets.append(tweet)
    df["Anonymized Tweet"] = anonymized_tweets

    # Replace the "Tweet Content" column with the "Anonymized Tweet" column
    df.drop(columns=["Tweet Content"], inplace=True)
    df.rename(columns={"Anonymized Tweet": "Tweet Content"}, inplace=True)

    # Show the final data table with the anonymized tweets in the "Tweet Content" column
    st.write("PII identified tweets")
    st.write(df)














    # k-anonymity
    # Identify the sensitive attributes
    sensitive_attributes = ['User Id', 'Name', 'Twitter Username', 'User Bio', 'Profile URL']

    # Determine the K value
    K = 3

    # Identify the quasi-identifiers
    quasi_identifiers = ['Tweet Location', 'User Followers', 'User Following', 'User Account Creation Date', 'Tweet Posted Time (UTC)']

    # Group the individuals based on the quasi-identifiers
    groups = df.groupby(quasi_identifiers)

    # Function to generalize Tweet Location to broader geographic regions
    def generalize_location(location):
            geolocator = Nominatim(user_agent="my-app")
            try:
                location = geolocator.geocode(location)
                if location is not None:
                    lat= location.raw.get('lat')
                    lon= location.raw.get('lon')
                    loc = lat + ',' + lon
                    country = geolocator.reverse(loc)
                    return country.raw.get('address').get('country')
                return "Unknown"
            except (GeocoderTimedOut, GeocoderServiceError):
                return "Unknown"


    # Function to generalize User Followers and User Following to ranges
    def generalize_follower_following(count):
        if count < 100:
            return '0-99'
        elif count >= 100 and count < 500:
            return '100-499'
        elif count >= 500 and count < 1000:
            return '500-999'
        else:
            return '1000+'

    # Function to generalize User Account Creation Date to month and year
    def generalize_account_creation_date(date):
        # Convert the input string to a datetime object
        dt_object = datetime.strptime(date, "%d-%m-%Y %H:%M")

        # Convert the datetime object to the desired format
        month_year = dt_object.strftime("%b, %Y")

        return month_year

    # Generalize or suppress the values of the quasi-identifiers in each group
    generalized_data = []
    for name, group in groups:
        # print(name, group)

        # st.write("name:")
        # st.write(name)

        # st.write("group:")
        # st.write(group)

        generalized_group = group.copy()


        for col in quasi_identifiers:
            if col == 'Tweet Location':
                # Generalize Tweet Location to broader geographic regions
                generalized_group[col] = group[col].apply(lambda x: generalize_location(x))
            elif col == 'User Followers' or col == 'User Following':
                # Generalize User Followers and User Following to ranges
                generalized_group[col] = group[col].apply(lambda x: generalize_follower_following(x))
            elif col == 'User Account Creation Date':
                # Generalize User Account Creation Date to month and year
                generalized_group[col] = group[col].apply(lambda x: generalize_account_creation_date(x))
            elif col == 'Tweet Posted Time (UTC)':
                # Generalize User Account Creation Date to month and year
                generalized_group[col] = group[col].apply(lambda x: generalize_account_creation_date(x))
        
        # st.write("generalized group:")
        # st.write(generalized_group)

        generalized_data.append(generalized_group)

    # Create a new DataFrame with the anonymized quasi-identifiers and the non-sensitive attribute Tweet Content
    anonymized_data = pd.concat(generalized_data)[['Tweet Location', 'User Followers', 'User Following', 'User Account Creation Date', 'Tweet Content', 'Tweet Posted Time (UTC)']]
    # st.write("anonymized data:")
    # st.write(anonymized_data)
    # # Verify that each group satisfies the K-Anonymity condition and merge groups if necessary
    k_anonymized_data = []
    for name, group in anonymized_data.groupby(quasi_identifiers):
        # st.write("name(k):")
        # st.write(name)
        # st.write("group(k):")
        # st.write(group)
        # st.write("length of group: ", len(group))
        if len(group) < K:
            # Merge the group with the nearest group that satisfies the K-Anonymity condition
            merged_group = pd.concat([group, anonymized_data.loc[anonymized_data.groupby(quasi_identifiers).groups[name]].iloc[0:K-len(group)]])
            k_anonymized_data.append(merged_group)
        else:
            k_anonymized_data.append(group)
        # st.write("k-anonymized data:")
        # st.write(k_anonymized_data)

    # Concatenate all the K-Anonymized groups into a final anonymized DataFrame
    df = pd.concat(k_anonymized_data)
    df = df.rename(columns={'Tweet Posted Time (UTC)': 'Tweet Posted Date'})
    # df = df.drop_duplicates()
    # st.write("K-Anonymized Dataset")
    # st.write(df)
